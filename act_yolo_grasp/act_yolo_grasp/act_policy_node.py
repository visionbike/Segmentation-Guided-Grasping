import os
import csv
import glob
import time
import pickle
import threading
import numpy as np
import cv2
import torch
import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from onnxruntime import InferenceSession
from openvino import CompiledModel

from .act import ACTPolicy


def _to_1d_np(x, name="array"):
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} cannot be reshaped to 1D, got shape={np.asarray(x).shape}")
    return arr


def _broadcast_stat(stat, target_dim, name):
    arr = _to_1d_np(stat, name=name)
    if arr.shape[0] == 1:
        arr = np.full((target_dim,), arr.item(), dtype=np.float32)
    if arr.shape[0] != target_dim:
        raise ValueError(f"{name} dim mismatch: got {arr.shape[0]}, expected {target_dim}")
    return arr


def _extract_state_dict(ckpt_obj):
    # Handle common checkpoint storage formats
    if isinstance(ckpt_obj, dict):
        for key in ["model_state_dict", "state_dict", "policy_state_dict", "model"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        # The project may itself be a state_dict
        tensor_like = any(torch.is_tensor(v) for v in ckpt_obj.values())
        if tensor_like:
            return ckpt_obj
    raise RuntimeError("Cannot extract state_dict from checkpoint")


def _strip_prefix_if_needed(state_dict):
    # Strip the common "module." prefix
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd


class ACTPolicyInferenceNode(Node):
    def __init__(self):
        super().__init__("act_policy_inference_node")

        # --------------------------------------------------
        # Parameters
        # --------------------------------------------------
        self.declare_parameter("device", "cpu")  # torch device; "cpu" on the Intel AI PC
        self.declare_parameter("freq", 10.0)

        self.declare_parameter("ckpt_dir", "")
        self.declare_parameter("ckpt_name", "policy_best.ckpt")

        # Inference backend:
        #   "openvino" - OpenVINO IR on Intel GPU/NPU/CPU (deploy default)
        #   "onnx"     - onnxruntime (verification fallback)
        #   "torch"    - original PyTorch checkpoint (A/B reference)
        self.declare_parameter("backend", "openvino")
        self.declare_parameter("ov_model_path", "")  # default: <ckpt_dir>/<ckpt stem>.xml
        self.declare_parameter("ov_device", "GPU")  # GPU | NPU | CPU | AUTO
        self.declare_parameter("onnx_path", "")  # default: <ckpt_dir>/<ckpt stem>.onnx
        self.declare_parameter("onnx_provider", "cpu")

        self.declare_parameter("observation_topic", "/motor_angle_feedback_topic")
        self.declare_parameter("action_topic", "/motor_action_angle_topic")
        self.declare_parameter("llm_state_topic", "/llm_state")

        # Whether to replay an observation CSV exported from Isaac Sim
        self.declare_parameter("use_dummy_observation", False)
        self.declare_parameter("dummy_observation_csv_path", "")
        self.declare_parameter("dummy_observation_loop", False)

        # camera names order MUST match training
        self.declare_parameter("camera_names", ["top", "left_wrist", "right_wrist"])

        # live image topics（only used when use_dummy_image=False)
        self.declare_parameter("image_topics", [
            "/top/YOLO_mask",
            "/left_wrist/YOLO_mask",
            "/right_wrist/YOLO_mask",
        ])

        # dummy image folders
        # order MUST match camera_names
        self.declare_parameter("use_dummy_images", False)
        self.declare_parameter("dummy_image_dirs", [
            "/tmp/cam0",
            "/tmp/cam1",
            "/tmp/cam2",
        ])
        self.declare_parameter("dummy_image_loop", False)

        # If training image has a fixed size, set this to the training size
        self.declare_parameter("image_resize_width", 320)
        self.declare_parameter("image_resize_height", 240)

        # ACT config
        self.declare_parameter("kl_weight", 10.0)
        self.declare_parameter("chunk_size", 10)
        self.declare_parameter("hidden_dim", 512)
        self.declare_parameter("dim_feedforward", 3200)
        self.declare_parameter("lr", 5e-5)
        self.declare_parameter("lr_backbone", 1e-5)
        self.declare_parameter("backbone", "resnet18")
        self.declare_parameter("enc_layers", 4)
        self.declare_parameter("dec_layers", 7)
        self.declare_parameter("nheads", 8)
        self.declare_parameter("sep_CNN", True)

        self.declare_parameter("temporal_ensemble", True)
        self.declare_parameter("ensemble_k", 0.05)

        # qpos / receiver dims
        self.declare_parameter("receiver_obs_dim", 17)
        self.declare_parameter("qpos_dim", 17)

        # logging（emtpy path = disable that CSV log)
        self.declare_parameter("log_actions_to_csv", True)
        self.declare_parameter("action_csv_path", "")
        self.declare_parameter("log_observations_to_csv", True)
        self.declare_parameter("observation_csv_path", "")

        # tunable subscriber QoS parameters
        self.declare_parameter("obs_qos_depth", 1)
        self.declare_parameter("image_qos_depth", 1)
        self.declare_parameter("obs_best_effort", True)
        self.declare_parameter("image_best_effort", True)
        self.declare_parameter("log_metrics_to_csv", True)
        self.declare_parameter("metric_csv_path", "")

        # --------------------------------------------------
        # Confidence assessment
        # --------------------------------------------------
        self.declare_parameter("confidence_enabled", True)
        self.declare_parameter("confidence_obs_start", 25)
        self.declare_parameter("confidence_obs_end", 35)
        self.declare_parameter("confidence_threshold", 0.050)
        self.declare_parameter("confidence_confirm_frames", 5)
        self.declare_parameter("confidence_recovery_steps", 40)

        self.confidence_enabled = bool(self.get_parameter("confidence_enabled").value)
        self.confidence_obs_start = int(self.get_parameter("confidence_obs_start").value)
        self.confidence_obs_end = int(self.get_parameter("confidence_obs_end").value)
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.confidence_confirm_frames = int(self.get_parameter("confidence_confirm_frames").value)
        self.confidence_recovery_steps = int(self.get_parameter("confidence_recovery_steps").value)

        # "observing"  → observation window after a task switch
        # "committed"  → judged successful; execution is no longer interrupted
        # "recovering" → returning to the initial position
        self.confidence_phase = "committed"  # start committed: the first task needs no assessment
        self.confidence_confirm_counter = 0
        self.confidence_recovery_step_count = 0  # step elapsed in the recovery phase

        self.log_metrics_to_csv = bool(self.get_parameter("log_metrics_to_csv").value)
        self.metric_csv_path = self.get_parameter("metric_csv_path").value

        self.device_str = str(self.get_parameter("device").value)
        self.freq = float(self.get_parameter("freq").value)
        self.ckpt_dir = str(self.get_parameter("ckpt_dir").value)
        self.ckpt_name = str(self.get_parameter("ckpt_name").value)
        self.backend = str(self.get_parameter("backend").value)
        self.ov_model_path = self.get_parameter("ov_model_path").value
        self.ov_device = str(self.get_parameter("ov_device").value)
        self.onnx_path = self.get_parameter("onnx_path").value
        self.onnx_provider = str(self.get_parameter("onnx_provider").value)
        self.observation_topic = self.get_parameter("observation_topic").value
        self.action_topic = self.get_parameter("action_topic").value
        self.llm_state_topic = self.get_parameter("llm_state_topic").value

        self.use_dummy_observation = bool(self.get_parameter("use_dummy_observation").value)
        self.dummy_observation_csv_path = self.get_parameter("dummy_observation_csv_path").value
        self.dummy_observation_loop = bool(self.get_parameter("dummy_observation_loop").value)

        self.camera_names = list(self.get_parameter("camera_names").value)
        self.image_topics = list(self.get_parameter("image_topics").value)

        self.use_dummy_images = bool(self.get_parameter("use_dummy_images").value)
        self.dummy_image_dirs = list(self.get_parameter("dummy_image_dirs").value)
        self.dummy_image_loop = bool(self.get_parameter("dummy_image_loop").value)

        self.image_resize_width = int(self.get_parameter("image_resize_width").value)
        self.image_resize_height = int(self.get_parameter("image_resize_height").value)
        self.kl_weight = float(self.get_parameter("kl_weight").value)
        self.chunk_size = int(self.get_parameter("chunk_size").value)
        self.hidden_dim = int(self.get_parameter("hidden_dim").value)
        self.dim_feedforward = int(self.get_parameter("dim_feedforward").value)
        self.lr = float(self.get_parameter("lr").value)
        self.lr_backbone = float(self.get_parameter("lr_backbone").value)
        self.backbone = str(self.get_parameter("backbone").value)
        self.enc_layers = int(self.get_parameter("enc_layers").value)
        self.dec_layers = int(self.get_parameter("dec_layers").value)
        self.nheads = int(self.get_parameter("nheads").value)
        self.sep_CNN = bool(self.get_parameter("sep_CNN").value)
        self.temporal_ensemble = bool(self.get_parameter("temporal_ensemble").value)
        self.ensemble_k = float(self.get_parameter("ensemble_k").value)
        self.receiver_obs_dim = int(self.get_parameter("receiver_obs_dim").value)
        self.qpos_dim = int(self.get_parameter("qpos_dim").value)
        self.log_actions_to_csv = bool(self.get_parameter("log_actions_to_csv").value)
        self.action_csv_path = self.get_parameter("action_csv_path").value
        self.log_observations_to_csv = bool(self.get_parameter("log_observations_to_csv").value)
        self.observation_csv_path = self.get_parameter("observation_csv_path").value

        self.obs_qos_depth = int(self.get_parameter("obs_qos_depth").value)
        self.image_qos_depth = int(self.get_parameter("image_qos_depth").value)
        self.obs_best_effort = bool(self.get_parameter("obs_best_effort").value)
        self.image_best_effort = bool(self.get_parameter("image_best_effort").value)

        if len(self.camera_names) != len(self.image_topics):
            raise ValueError(
                f"camera_names length ({len(self.camera_names)}) != image_topics length ({len(self.image_topics)})"
            )

        if len(self.camera_names) != len(self.dummy_image_dirs):
            raise ValueError(
                f"camera_names length ({len(self.camera_names)}) != dummy_image_dirs length ({len(self.dummy_image_dirs)})"
            )

        if self.ckpt_dir == "":
            raise ValueError("ckpt_dir is empty")

        self.steps_since_switch = 0
        self.switch_count = 0
        self.obs_window_start_step = 0  # start of the current observation window

        # --------------------------------------------------
        # Ensemble std statistics per state
        # --------------------------------------------------
        self.ensemble_std_state_values = []
        self.ensemble_std_window_values = []
        self.ensemble_std_state_mean = float("nan")
        self.ensemble_std_window_mean = float("nan")

        # --------------------------------------------------
        # callback groups / lock
        # --------------------------------------------------
        # sensor subscriber in a Reentrant group, timer in a MutuallyExclusive group
        self.sensor_group = ReentrantCallbackGroup()
        self.control_group = MutuallyExclusiveCallbackGroup()
        self.data_lock = threading.Lock()

        # --------------------------------------------------
        # Set device
        # --------------------------------------------------
        if self.device_str.startswith("cuda") and not torch.cuda.is_available():
            self.get_logger().warning("CUDA requested but not available. Fallback to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.device_str)

        # display label built from declared attrs (.type / .index) - torch's
        # stubs declare no __str__ on torch.device, which trips IDE inspections
        self.device_label = (
            self.device.type if self.device.index is None
            else f"{self.device.type}:{self.device.index}"
        )

        # --------------------------------------------------
        # Load stats + policy
        # --------------------------------------------------
        self.stats = self.load_dataset_stats()
        self.policy: ACTPolicy | None = None
        self.onnx_session: InferenceSession | None = None
        self.ov_compiled: CompiledModel | None = None
        if self.backend == "openvino":
            self.ov_compiled = self.load_openvino_policy()
        elif self.backend == "onnx":
            self.onnx_session = self.load_onnx_policy()
        elif self.backend == "torch":
            self.policy = self.load_act_policy()
        else:
            raise ValueError(f"Unknown backend: {self.backend} (openvino|onnx|torch)")

        self.qpos_mean = _broadcast_stat(self.stats["qpos_mean"], self.qpos_dim, "qpos_mean")
        self.qpos_std = _broadcast_stat(self.stats["qpos_std"], self.qpos_dim, "qpos_std")

        action_mean_raw = _to_1d_np(self.stats["action_mean"], "action_mean")
        action_std_raw = _to_1d_np(self.stats["action_std"], "action_std")

        if action_mean_raw.shape[0] == 1:
            self.action_mean = action_mean_raw
            self.action_std = action_std_raw
            self.action_dim = None
        else:
            self.action_mean = action_mean_raw
            self.action_std = action_std_raw
            self.action_dim = action_mean_raw.shape[0]

        # --------------------------------------------------
        # Internal state
        # --------------------------------------------------
        self.latest_obs = None
        self.latest_obs_stamp = None

        # --------------------------------------------------
        # LLM state machine
        # --------------------------------------------------
        # action order:
        # [L1..L7, R1..R7, L_gripper, R_gripper, Neck]
        self.initial_action = np.asarray([
            np.deg2rad(35.0), 0.0, 0.0, np.deg2rad(-120.0), 0.0, 0.0, 0.0,
            np.deg2rad(-35.0), 0.0, 0.0, np.deg2rad(120.0), 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ], dtype=np.float32)

        # /llm_state format: [t0,t1,t2, d0,d1,d2]  bits 0-2=target, bits 3-5=destination
        # to add new states later, only this dictionary needs changing
        self.TARGETS = ["red_cookies", "green_tea", "yellow_cookies"]
        self.DESTINATIONS = ["mid_basket", "right_basket", "human_hand"]

        self.llm_modes = {
            "all_gray": {
                "type": "fixed_pose",
                "action": self.initial_action,
            },
            "red_cookies_to_mid_basket": {"type": "act_policy"},
            "red_cookies_to_right_basket": {"type": "act_policy"},
            "green_tea_to_mid_basket": {"type": "act_policy"},
            "green_tea_to_right_basket": {"type": "act_policy"},
            "yellow_cookies_to_mid_basket": {"type": "act_policy"},
            "red_cookies_to_human_hand": {"type": "act_policy"},
            "green_tea_to_human_hand": {"type": "act_policy"},
            "yellow_cookies_to_human_hand": {"type": "act_policy"},
        }

        self.latest_llm_state = np.zeros(
            len(self.TARGETS) + len(self.DESTINATIONS), dtype=np.float32
        )
        self.current_llm_mode = "all_gray"

        self.step = 0
        self.start_time = time.time()

        # chunk cache / ensemble
        self.cached_all_actions = None
        self.pred_history = []

        # dummy observation CSV
        self.dummy_obs_data = []
        self.required_obs_columns = [f"q{i}" for i in range(self.qpos_dim)]
        if self.use_dummy_observation:
            self.load_dummy_observation_csv()

        # dummy images
        self.dummy_image_paths = {cam_name: [] for cam_name in self.camera_names}
        if self.use_dummy_images:
            self.load_dummy_image_dirs()

        # also record a timestamp per camera for later freshness checks
        self.bridge = CvBridge()
        self.latest_images: dict[str, np.ndarray | None] = {name: None for name in self.camera_names}
        self.latest_image_stamps: dict[str, object] = {name: None for name in self.camera_names}
        self.image_subs = []

        # --------------------------------------------------
        # ROS interfaces
        # --------------------------------------------------
        self.action_pub = self.create_publisher(Float32MultiArray, self.action_topic, 10)

        self.ensemble_std_window_mean_pub = self.create_publisher(
            Float32MultiArray,
            "/ensemble_std_window_mean",
            10
        )

        # high-rate observation：keep only the latest message to avoid queue delay
        obs_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, self.obs_qos_depth),
            reliability=ReliabilityPolicy.BEST_EFFORT if self.obs_best_effort else ReliabilityPolicy.RELIABLE,
        )

        self.observation_sub = self.create_subscription(
            Float32MultiArray,
            self.observation_topic,
            self.observation_callback,
            obs_qos,
            callback_group=self.sensor_group,
        )

        self.llm_state_sub = self.create_subscription(
            Float32MultiArray,
            self.llm_state_topic,
            self.llm_state_callback,
            10,
            callback_group=self.sensor_group,
        )

        if not self.use_dummy_images:
            # same for high-rate images: keep only the latest frame
            image_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=max(1, self.image_qos_depth),
                reliability=ReliabilityPolicy.BEST_EFFORT if self.image_best_effort else ReliabilityPolicy.RELIABLE,
            )

            for cam_name, topic in zip(self.camera_names, self.image_topics):
                sub = self.create_subscription(
                    Image,
                    topic,
                    self.make_image_callback(cam_name),
                    image_qos,
                    callback_group=self.sensor_group,
                )
                self.image_subs.append(sub)

        # --------------------------------------------------
        # CSV logging
        # --------------------------------------------------
        self.action_csv_file = None
        self.action_csv_writer = None
        self.action_csv_initialized = False

        self.obs_csv_file = None
        self.metric_csv_file = None
        self.metric_csv_writer = None
        self.metric_csv_initialized = False
        self.obs_csv_writer = None
        self.obs_csv_initialized = False

        if self.log_actions_to_csv:
            self.init_action_csv()
        if self.log_observations_to_csv:
            self.init_observation_csv()
        if self.log_metrics_to_csv:
            self.init_metric_csv()

        # --------------------------------------------------
        # Timer
        # --------------------------------------------------
        timer_period = 1.0 / self.freq
        self.timer = self.create_timer(
            timer_period,
            self.timer_callback,
            callback_group=self.control_group,
        )

        self.get_logger().info("=========================================")
        self.get_logger().info("ACT policy inference node started")
        self.get_logger().info(f"device                    : {self.device_label}")
        self.get_logger().info(f"freq                      : {self.freq} Hz")
        self.get_logger().info(f"kl_weight                 : {self.kl_weight}")
        self.get_logger().info(f"sep_CNN                   : {self.sep_CNN}")
        self.get_logger().info(f"ckpt_dir                  : {self.ckpt_dir}")
        self.get_logger().info(f"ckpt_name                 : {self.ckpt_name}")
        self.get_logger().info(f"backend                   : {self.backend}")
        self.get_logger().info(f"ov_model_path             : {self.ov_model_path}")
        self.get_logger().info(f"ov_device                 : {self.ov_device}")
        self.get_logger().info(f"onnx_path                 : {self.onnx_path}")
        self.get_logger().info(f"onnx_provider             : {self.onnx_provider}")
        self.get_logger().info(f"observation_topic         : {self.observation_topic}")
        self.get_logger().info(f"action_topic              : {self.action_topic}")
        self.get_logger().info(f"llm_state_topic           : {self.llm_state_topic}")
        self.get_logger().info(f"use_dummy_observation     : {self.use_dummy_observation}")
        self.get_logger().info(f"dummy_observation_csv     : {self.dummy_observation_csv_path}")
        self.get_logger().info(f"dummy_observation_loop    : {self.dummy_observation_loop}")
        self.get_logger().info(f"use_dummy_images          : {self.use_dummy_images}")
        self.get_logger().info(f"dummy_image_dirs          : {self.dummy_image_dirs}")
        self.get_logger().info(f"dummy_image_loop         : {self.dummy_image_loop}")
        self.get_logger().info(f"camera_names              : {self.camera_names}")
        self.get_logger().info(f"image_topics              : {self.image_topics}")
        self.get_logger().info(f"image_resize              : {self.image_resize_width} x {self.image_resize_height}")
        self.get_logger().info(f"qpos_dim                  : {self.qpos_dim}")
        self.get_logger().info(f"chunk_size                : {self.chunk_size}")
        self.get_logger().info(f"temporal_ensemble         : {self.temporal_ensemble}")
        self.get_logger().info(f"ensemble_k                : {self.ensemble_k}")
        self.get_logger().info(f"obs_qos_depth             : {self.obs_qos_depth}")
        self.get_logger().info(f"image_qos_depth           : {self.image_qos_depth}")
        self.get_logger().info(f"obs_best_effort           : {self.obs_best_effort}")
        self.get_logger().info(f"image_best_effort         : {self.image_best_effort}")
        self.get_logger().info(f"log_actions_to_csv        : {self.log_actions_to_csv}")
        self.get_logger().info(f"log_observations_to_csv   : {self.log_observations_to_csv}")
        self.get_logger().info("=========================================")

    # --------------------------------------------------
    # Load dataset stats
    # --------------------------------------------------
    def load_dataset_stats(self):
        stats_path = os.path.join(self.ckpt_dir, "dataset_stats.pkl")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"dataset_stats.pkl not found: {stats_path}")

        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        required_keys = ["qpos_mean", "qpos_std", "action_mean", "action_std"]
        for k in required_keys:
            if k not in stats:
                raise KeyError(f"dataset_stats.pkl missing key: {k}")

        self.get_logger().info(f"Loaded dataset stats from: {stats_path}")
        return stats

    # --------------------------------------------------
    # Load ACT policy
    # --------------------------------------------------
    def load_act_policy(self):
        ckpt_path = os.path.join(self.ckpt_dir, self.ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ACT checkpoint not found: {ckpt_path}")

        policy_config = {
            "lr": self.lr,
            "num_queries": self.chunk_size,
            "kl_weight": self.kl_weight,
            "hidden_dim": self.hidden_dim,
            "dim_feedforward": self.dim_feedforward,
            "lr_backbone": self.lr_backbone,
            "backbone": self.backbone,
            "enc_layers": self.enc_layers,
            "dec_layers": self.dec_layers,
            "nheads": self.nheads,
            "camera_names": self.camera_names,
            "sep_CNN": self.sep_CNN,
            "obs_dim": 17,
            "action_dim": 17,
        }

        policy = ACTPolicy(policy_config)

        ckpt_obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = _extract_state_dict(ckpt_obj)
        state_dict = _strip_prefix_if_needed(state_dict)

        load_status = policy.load_state_dict(state_dict, strict=True)
        self.get_logger().info(f"Loaded ACT ckpt: {ckpt_path}")
        self.get_logger().info(f"load_state_dict status: {load_status}")

        policy.to(self.device)
        policy.eval()
        return policy

    # --------------------------------------------------
    # Load ONNX policy
    # --------------------------------------------------
    def load_onnx_policy(self):
        onnx_path = self.onnx_path or os.path.join(
            self.ckpt_dir, os.path.splitext(self.ckpt_name)[0] + ".onnx"
        )
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        import onnxruntime as ort

        if self.onnx_provider == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        session = ort.InferenceSession(onnx_path, providers=providers)
        actual_providers = list(session.get_providers())

        self.get_logger().info(f"Loaded ONNX model: {onnx_path}")
        self.get_logger().info(f"Requested providers: {providers}, actual providers: {actual_providers}")

        if self.onnx_provider == "cuda" and "CUDAExecutionProvider" not in actual_providers:
            self.get_logger().warning(
                "CUDAExecutionProvider not available (onnxruntime-gpu not installed, or no usable "
                "CUDA device/cuDNN for onnxruntime). Falling back to CPUExecutionProvider - "
                "inference will be significantly slower."
            )

        return session

    # --------------------------------------------------
    # Load OpenVINO policy (Intel Arc GPU / NPU / CPU)
    # --------------------------------------------------
    def load_openvino_policy(self):
        import openvino as ov

        # default: <ckpt_dir>/<ckpt stem>_openvino_model/ (folder layout, like YOLO)
        model_path = self.ov_model_path or os.path.join(
            self.ckpt_dir, os.path.splitext(self.ckpt_name)[0] + "_openvino_model"
        )
        # accept either an IR folder (glob the .xml inside) or a direct .xml file
        if os.path.isdir(model_path):
            xmls = sorted(glob.glob(os.path.join(model_path, "*.xml")))
            if not xmls:
                raise FileNotFoundError(f"No .xml found in OpenVINO IR folder: {model_path}")
            xml_path = xmls[0]
        else:
            xml_path = model_path if model_path.endswith(".xml") else model_path + ".xml"
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"OpenVINO IR not found: {xml_path}")

        core = ov.Core()
        model = core.read_model(xml_path)

        # Static batch-1 shapes: required for NPU, better kernels on GPU/CPU.
        model.reshape({
            "qpos": [1, self.qpos_dim],
            "image": [1, len(self.camera_names), 3,
                      self.image_resize_height, self.image_resize_width],
        })

        available = core.available_devices
        compiled = core.compile_model(model, self.ov_device,
                                      {"PERFORMANCE_HINT": "LATENCY"})

        self.get_logger().info(f"Loaded OpenVINO IR: {xml_path}")
        self.get_logger().info(
            f"ov_device={self.ov_device}, available_devices={available}"
        )
        if self.ov_device not in ("AUTO", "CPU") and self.ov_device not in available:
            self.get_logger().warning(
                f"Requested OpenVINO device '{self.ov_device}' not in {available} - "
                f"check intel-compute-runtime / linux-npu-driver installation."
            )
        return compiled

    # --------------------------------------------------
    # CSV logging
    # --------------------------------------------------
    def init_action_csv(self):
        if not self.action_csv_path:
            self.log_actions_to_csv = False
            self.get_logger().warning("action_csv_path empty - action CSV logging disabled")
            return
        parent_dir = os.path.dirname(self.action_csv_path)
        if parent_dir != "":
            os.makedirs(parent_dir, exist_ok=True)
        self.action_csv_file = open(self.action_csv_path, "w", newline="", encoding="utf-8")
        self.get_logger().info(f"Action CSV logging enabled: {self.action_csv_path}")

    def init_observation_csv(self):
        if not self.observation_csv_path:
            self.log_observations_to_csv = False
            self.get_logger().warning("observation_csv_path empty - observation CSV logging disabled")
            return
        parent_dir = os.path.dirname(self.observation_csv_path)
        if parent_dir != "":
            os.makedirs(parent_dir, exist_ok=True)
        self.obs_csv_file = open(self.observation_csv_path, "w", newline="", encoding="utf-8")
        self.get_logger().info(f"Observation CSV logging enabled: {self.observation_csv_path}")

    def init_metric_csv(self):
        if not self.metric_csv_path:
            self.log_metrics_to_csv = False
            self.get_logger().warning("metric_csv_path empty - metric CSV logging disabled")
            return
        parent_dir = os.path.dirname(self.metric_csv_path)
        if parent_dir != "":
            os.makedirs(parent_dir, exist_ok=True)
        self.metric_csv_file = open(self.metric_csv_path, "w", newline="", encoding="utf-8")
        self.metric_csv_writer = csv.writer(self.metric_csv_file)
        self.metric_csv_writer.writerow([
            "step", "time_sec", "mode",
            "switch_count", "steps_since_switch",
            "ensemble_std",
            "ensemble_std_state_mean",
            "ensemble_std_window_mean",
            "ensemble_std_state_count",
            "ensemble_n",
            "confidence_phase",
            "confidence_confirm_counter",
            "zscore_max", "zscore_mean",
        ])
        self.metric_csv_file.flush()
        self.metric_csv_initialized = True
        self.get_logger().info(f"Metric CSV logging enabled: {self.metric_csv_path}")

    def log_metrics(self, mode, ensemble_std, ensemble_n, zscore_max, zscore_mean):
        if not self.log_metrics_to_csv or self.metric_csv_file is None:
            return

        timestamp = time.time() - self.start_time

        self.metric_csv_writer.writerow([
            self.step,
            f"{timestamp:.4f}",
            mode,
            self.switch_count,
            self.steps_since_switch,

            f"{ensemble_std:.6f}" if ensemble_std is not None else "nan",
            f"{self.ensemble_std_state_mean:.6f}" if not np.isnan(self.ensemble_std_state_mean) else "nan",
            f"{self.ensemble_std_window_mean:.6f}" if not np.isnan(self.ensemble_std_window_mean) else "nan",
            len(self.ensemble_std_state_values),

            ensemble_n,
            self.confidence_phase,
            self.confidence_confirm_counter,

            f"{zscore_max:.6f}",
            f"{zscore_mean:.6f}",
        ])

        self.metric_csv_file.flush()

    def update_ensemble_std_stats(self, ensemble_std):
        if ensemble_std is None:
            return

        ensemble_std = float(ensemble_std)

        # mean over current state
        self.ensemble_std_state_values.append(ensemble_std)
        self.ensemble_std_state_mean = float(np.mean(self.ensemble_std_state_values))

        # mean only inside confidence observation window
        steps_in_window = self.steps_since_switch - self.obs_window_start_step
        if self.confidence_obs_start <= steps_in_window <= self.confidence_obs_end:
            self.ensemble_std_window_values.append(ensemble_std)
            self.ensemble_std_window_mean = float(np.mean(self.ensemble_std_window_values))

        if self.confidence_phase == "observing":
            msg = Float32MultiArray()
            msg.data = [
                float(self.ensemble_std_window_mean) if not np.isnan(self.ensemble_std_window_mean) else float("nan")]
            self.ensemble_std_window_mean_pub.publish(msg)

    def write_action_csv_header_if_needed(self, action_dim):
        if self.action_csv_initialized or self.action_csv_file is None:
            return
        header = ["step", "time_sec"] + [f"action_{i}" for i in range(action_dim)]
        self.action_csv_writer = csv.writer(self.action_csv_file)
        self.action_csv_writer.writerow(header)
        self.action_csv_file.flush()
        self.action_csv_initialized = True

    def write_obs_csv_header_if_needed(self, obs_dim):
        if self.obs_csv_initialized or self.obs_csv_file is None:
            return
        header = ["step", "time_sec"] + [f"obs_{i}" for i in range(obs_dim)]
        self.obs_csv_writer = csv.writer(self.obs_csv_file)
        self.obs_csv_writer.writerow(header)
        self.obs_csv_file.flush()
        self.obs_csv_initialized = True

    def log_actions(self, actions_np):
        if not self.log_actions_to_csv or self.action_csv_file is None:
            return
        self.write_action_csv_header_if_needed(len(actions_np))
        timestamp = time.time() - self.start_time
        row = [self.step, timestamp] + actions_np.astype(np.float32).tolist()
        self.action_csv_writer.writerow(row)
        self.action_csv_file.flush()

    def log_observations(self, obs_np):
        if not self.log_observations_to_csv or self.obs_csv_file is None:
            return
        self.write_obs_csv_header_if_needed(len(obs_np))
        timestamp = time.time() - self.start_time
        row = [self.step, timestamp] + obs_np.astype(np.float32).tolist()
        self.obs_csv_writer.writerow(row)
        self.obs_csv_file.flush()

    # --------------------------------------------------
    # Dummy observation CSV
    # --------------------------------------------------
    def load_dummy_observation_csv(self):
        if self.dummy_observation_csv_path == "":
            raise ValueError("use_dummy_observation=True but dummy_observation_csv_path is empty")

        if not os.path.exists(self.dummy_observation_csv_path):
            raise FileNotFoundError(f"dummy observation csv not found: {self.dummy_observation_csv_path}")

        try:
            with open(self.dummy_observation_csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)

                if reader.fieldnames is None:
                    raise ValueError("CSV has no header")

                fieldnames = list(reader.fieldnames)
                missing_cols = [c for c in self.required_obs_columns if c not in fieldnames]
                if len(missing_cols) > 0:
                    raise ValueError(
                        f"CSV missing required columns: {missing_cols}\n"
                        f"Current columns: {fieldnames}"
                    )

                parsed_rows = []
                for row_idx, row in enumerate(reader, start=2):
                    try:
                        obs = np.asarray(
                            [float(row[col]) for col in self.required_obs_columns],
                            dtype=np.float32
                        )
                    except Exception as e:
                        raise ValueError(f"Failed to parse CSV row {row_idx}: {e}")

                    if obs.shape[0] != self.qpos_dim:
                        raise ValueError(
                            f"CSV row {row_idx} observation dim is {obs.shape[0]}, expected {self.qpos_dim}"
                        )

                    parsed_rows.append(obs)

            if len(parsed_rows) == 0:
                raise ValueError("No valid observation rows found in CSV")

            self.dummy_obs_data = parsed_rows

            self.get_logger().info(
                f"Loaded {len(self.dummy_obs_data)} dummy observations from: {self.dummy_observation_csv_path}"
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load dummy observation CSV: {e}")

    # --------------------------------------------------
    # Dummy image folders
    # --------------------------------------------------
    def load_dummy_image_dirs(self):
        valid_exts = {".png", ".jpg", ".jpeg"}

        for cam_name, img_dir in zip(self.camera_names, self.dummy_image_dirs):
            if not os.path.isdir(img_dir):
                raise FileNotFoundError(f"dummy image dir not found for {cam_name}: {img_dir}")

            files = []
            for fname in os.listdir(img_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext in valid_exts:
                    files.append(os.path.join(img_dir, fname))

            files = sorted(files)

            if len(files) == 0:
                raise ValueError(f"No image files found in {img_dir} for camera {cam_name}")

            self.dummy_image_paths[cam_name] = files
            self.get_logger().info(
                f"Loaded {len(files)} images for {cam_name} from: {img_dir}"
            )

    @staticmethod
    def _resolve_index(
            data_index: int,
        total_len: int,
        loop: bool,
        source_name: str
    ) -> tuple[int | None, str | None]:
        if total_len <= 0:
            return None, f"{source_name} is empty"

        if data_index < total_len:
            return data_index, None

        if loop:
            return data_index % total_len, None

        return None, f"{source_name} reached end at index {data_index}, total_len={total_len}"

    # --------------------------------------------------
    # Observation source
    # --------------------------------------------------
    def get_current_observation(self, data_index):
        if self.use_dummy_observation:
            idx, err = self._resolve_index(
                data_index,
                len(self.dummy_obs_data),
                self.dummy_observation_loop,
                "Dummy observation CSV",
            )
            if err is not None:
                return None, err
            assert idx is not None
            return self.dummy_obs_data[idx].copy(), None

        with self.data_lock:
            if self.latest_obs is None:
                return None, "No joint observation received yet"
            return self.latest_obs.copy(), None

    # --------------------------------------------------
    # Image source
    # --------------------------------------------------
    def get_current_images(self, data_index):
        if self.use_dummy_images:
            img_dict = {}
            for cam_name in self.camera_names:
                img_list = self.dummy_image_paths[cam_name]
                idx, err = self._resolve_index(
                    data_index,
                    len(img_list),
                    self.dummy_image_loop,
                    f"Dummy image folder for {cam_name}",
                )
                if err is not None:
                    return None, err
                assert idx is not None

                img_path = img_list[idx]
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    return None, f"Failed to read image: {img_path}"

                img_dict[cam_name] = img

            return img_dict, None

        with self.data_lock:
            missing_cams = [name for name in self.camera_names if self.latest_images[name] is None]
            if len(missing_cams) > 0:
                return None, f"Missing camera images: {missing_cams}"

            return {k: v.copy() for k, v in self.latest_images.items() if v is not None}, None

    def parse_llm_state_to_mode(self, data: np.ndarray) -> str:
        expected_dim = len(self.TARGETS) + len(self.DESTINATIONS)  # 6
        if data.shape[0] != expected_dim:
            self.get_logger().warning(
                f"[/llm_state] dimension mismatch: got {data.shape[0]}, expected {expected_dim}"
            )
            return self.current_llm_mode

        if np.max(data) < 0.5:
            return "all_gray"

        if np.min(data) >= 0.5:  # full_reset [1,1,1,1,1,1] → treat as all_gray
            return "all_gray"

        target = data[:len(self.TARGETS)]
        dest = data[len(self.TARGETS):]

        target_name = self.TARGETS[int(np.argmax(target))]
        dest_name = self.DESTINATIONS[int(np.argmax(dest))]
        mode = f"{target_name}_to_{dest_name}"

        if mode not in self.llm_modes:
            self.get_logger().warning(
                f"[LLM] Decoded invalid mode: {mode}, fallback to all_gray"
            )
            return "all_gray"

        return mode

    def llm_state_callback(self, msg: Float32MultiArray):
        data = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        new_mode = self.parse_llm_state_to_mode(data)

        with self.data_lock:
            old_mode = self.current_llm_mode
            self.latest_llm_state = data.copy()
            self.current_llm_mode = new_mode

            if old_mode != new_mode:
                self.cached_all_actions = None
                self.pred_history = []
                self.steps_since_switch = 0
                self.switch_count += 1
                self.ensemble_std_state_values = []
                self.ensemble_std_window_values = []
                self.ensemble_std_state_mean = float("nan")
                self.ensemble_std_window_mean = float("nan")

                mode_type = self.llm_modes.get(new_mode, {}).get("type", "act_policy")
                if mode_type == "fixed_pose":
                    self.confidence_phase = "committed"
                else:
                    self.confidence_phase = "observing"
                self.confidence_confirm_counter = 0
                self.confidence_recovery_step_count = 0
                self.obs_window_start_step = 0

        if old_mode != new_mode:
            self.get_logger().info(
                f"[LLM STATE] {old_mode} -> {new_mode}, llm_state={data.tolist()}"
            )

    def publish_fixed_action(self, action_np, mode_name):
        action_np = np.asarray(action_np, dtype=np.float32).reshape(-1)

        msg = Float32MultiArray()
        msg.data = action_np.tolist()
        self.action_pub.publish(msg)

        self.log_actions(action_np)

        self.get_logger().info(
            f"step={self.step}, mode={mode_name}, publish fixed action"
        )

    # --------------------------------------------------
    # Observation  (dimension remap)
    # --------------------------------------------------
    def observation_callback(self, msg: Float32MultiArray):
        data = np.asarray(msg.data, dtype=np.float32).reshape(-1)

        if data.shape[0] != self.receiver_obs_dim:
            self.get_logger().warning(
                f"Observation dimension mismatch: got {data.shape[0]}, expected {self.receiver_obs_dim}"
            )
            return

        # Assumed format:
        # [R1...R7, L1...L7, R_gripper, L_gripper, Neck]
        # unit: degree
        if data.shape[0] != 17:
            self.get_logger().warning(
                f"Expected 17-dim real observation, but got {data.shape[0]}"
            )
            return

        R1, R2, R3, R4, R5, R6, R7 = data[0:7]
        L1, L2, L3, L4, L5, L6, L7 = data[7:14]
        R_Gripper = data[14]
        L_Gripper = data[15]
        Neck = data[16]

        obs_17 = np.asarray([
            L1, L2, L3, L4, L5, L6, L7,
            R1, R2, R3, R4, R5, R6, R7,
            L_Gripper, R_Gripper, Neck
        ], dtype=np.float32)

        if obs_17.shape[0] != self.qpos_dim:
            self.get_logger().error(
                f"Built qpos dim mismatch: got {obs_17.shape[0]}, expected {self.qpos_dim}"
            )
            return

        with self.data_lock:
            self.latest_obs = obs_17
            self.latest_obs_stamp = self.get_clock().now()

    # --------------------------------------------------
    # Image callback factory
    # --------------------------------------------------
    def make_image_callback(self, cam_name):
        def _callback(msg: Image):
            try:
                enc = msg.encoding.lower()

                # convert to an OpenCV-friendly format where possible
                if "mono" in enc or enc in ["8uc1", "16uc1"]:
                    img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
                elif enc == "rgb8":
                    img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif enc == "rgba8":
                    img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgba8")
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
                else:
                    img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

                img = np.asarray(img)

                with self.data_lock:
                    self.latest_images[cam_name] = img
                    self.latest_image_stamps[cam_name] = self.get_clock().now()

            except Exception as e:
                self.get_logger().error(f"Failed to convert image for {cam_name}: {e}")

        return _callback

    # --------------------------------------------------
    # Preprocess one image to [C,H,W], float in [0,1]
    # --------------------------------------------------
    def preprocess_one_image(self, img):
        if img is None:
            raise ValueError("input image is None")

        img = np.asarray(img)

        # resize if requested
        if self.image_resize_width > 0 and self.image_resize_height > 0:
            if img.shape[1] != self.image_resize_width or img.shape[0] != self.image_resize_height:
                img = cv2.resize(
                    img,
                    (self.image_resize_width, self.image_resize_height),
                    interpolation=cv2.INTER_NEAREST if len(img.shape) == 2 else cv2.INTER_LINEAR
                )

        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)

        if img.ndim != 3:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        if img.shape[2] == 4:
            img = img[:, :, :3]

        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported channel count: {img.shape[2]}")

        img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        img = np.clip(img, 0.0, 1.0)
        chw = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(chw).float()

    # --------------------------------------------------
    # Build ACT inputs
    # --------------------------------------------------
    def build_act_inputs(self):
        # use the same step index to align observation row and image frame
        data_index = self.step

        qpos_source, obs_err = self.get_current_observation(data_index)
        if obs_err is not None:
            return None, None, obs_err

        image_source, img_err = self.get_current_images(data_index)
        if img_err is not None:
            return None, None, img_err

        qpos_np = qpos_source.astype(np.float32).reshape(-1)

        if qpos_np.shape[0] != self.qpos_dim:
            return None, None, f"qpos dim mismatch: got {qpos_np.shape[0]}, expected {self.qpos_dim}"

        eps = 1e-8
        qpos_norm = (qpos_np - self.qpos_mean) / np.maximum(self.qpos_std, eps)
        qpos = torch.from_numpy(qpos_norm).float().to(self.device).unsqueeze(0)

        img_tensors = []
        for cam_name in self.camera_names:
            img_tensor = self.preprocess_one_image(image_source[cam_name])
            img_tensors.append(img_tensor)

        curr_image = torch.stack(img_tensors, dim=0).unsqueeze(0).to(self.device)

        return qpos, curr_image, None

    # --------------------------------------------------
    # ACT forward
    # --------------------------------------------------
    def query_policy_chunk(self, qpos, curr_image):
        if self.backend == "openvino":
            assert self.ov_compiled is not None
            qpos_np = qpos.detach().cpu().numpy().astype(np.float32)
            image_np = curr_image.detach().cpu().numpy().astype(np.float32)
            out_np = self.ov_compiled({"qpos": qpos_np, "image": image_np})["action"]
            out = torch.from_numpy(np.asarray(out_np)).to(self.device)
        elif self.backend == "onnx":
            assert self.onnx_session is not None
            qpos_np = qpos.detach().cpu().numpy().astype(np.float32)
            image_np = curr_image.detach().cpu().numpy().astype(np.float32)
            out_np = self.onnx_session.run(["action"], {"qpos": qpos_np, "image": image_np})[0]
            out = torch.from_numpy(out_np).to(self.device)
        elif self.backend == "torch":
            with torch.inference_mode():
                assert self.policy is not None
                out = self.policy(qpos, curr_image)
                assert isinstance(out, torch.Tensor)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        # if not torch.is_tensor(out):
        #     raise TypeError(f"policy output must be torch.Tensor, got {type(out)}")

        if out.ndim != 3:
            raise ValueError(f"Expected ACT output ndim=3, got shape={tuple(out.shape)}")

        if out.shape[0] != 1:
            raise ValueError(f"Expected ACT batch size 1, got shape={tuple(out.shape)}")

        if out.shape[1] != self.chunk_size:
            raise ValueError(
                f"Expected num_queries/chunk_size={self.chunk_size}, got shape={tuple(out.shape)}"
            )

        return out  # [1, num_queries, action_dim]

    # --------------------------------------------------
    # Denormalize ACT action
    # --------------------------------------------------
    def post_process_action(self, raw_action_np):
        raw_action_np = np.asarray(raw_action_np, dtype=np.float32).reshape(-1)

        if self.action_dim is None:
            self.action_dim = raw_action_np.shape[0]
            self.action_mean = _broadcast_stat(self.stats["action_mean"], self.action_dim, "action_mean")
            self.action_std = _broadcast_stat(self.stats["action_std"], self.action_dim, "action_std")

        if raw_action_np.shape[0] != self.action_dim:
            raise ValueError(
                f"Action dim mismatch: got {raw_action_np.shape[0]}, expected {self.action_dim}"
            )

        return raw_action_np * self.action_std + self.action_mean

    # --------------------------------------------------
    # Compute one action
    # --------------------------------------------------
    def compute_action(self, qpos, curr_image):
        if not self.temporal_ensemble:
            if self.cached_all_actions is None or (self.step % self.chunk_size == 0):
                self.cached_all_actions = self.query_policy_chunk(qpos, curr_image)
            query_index = self.step % self.chunk_size
            raw_action = self.cached_all_actions[0, query_index, :].detach().cpu().numpy()
            action = self.post_process_action(raw_action)
            return action, None, 0  # ← non-ensemble mode

        all_actions = self.query_policy_chunk(qpos, curr_image)[0].detach()
        self.pred_history.append((self.step, all_actions))
        self.pred_history = [
            (s, pred) for (s, pred) in self.pred_history
            if (self.step - s) < self.chunk_size
        ]

        actions_for_now = []
        for pred_step, pred_chunk in self.pred_history:
            offset = self.step - pred_step
            if 0 <= offset < pred_chunk.shape[0]:
                actions_for_now.append(pred_chunk[offset])

        if len(actions_for_now) == 0:
            raise RuntimeError("No valid temporal ensemble action found")

        actions_for_now = torch.stack(actions_for_now, dim=0).to(self.device)

        ensemble_n = actions_for_now.shape[0]
        ensemble_std = actions_for_now.std(dim=0).mean().item() if ensemble_n > 1 else None

        n = actions_for_now.shape[0]
        weights = np.exp(self.ensemble_k * np.arange(n, dtype=np.float32))
        weights = weights / weights.sum()
        weights = torch.from_numpy(weights).to(self.device).unsqueeze(1)

        raw_action = (actions_for_now * weights).sum(dim=0).detach().cpu().numpy()
        action = self.post_process_action(raw_action)
        return action, ensemble_std, ensemble_n  # ← also return std and n

    def assess_confidence(self, ensemble_std):
        if not self.confidence_enabled:
            self.confidence_phase = "committed"
            return

        if self.confidence_phase == "committed":
            return

        if self.confidence_phase == "recovering":
            return  # recovering is handled timer_callback

        # --- observing phase ---
        steps_in_window = self.steps_since_switch - self.obs_window_start_step

        if steps_in_window < self.confidence_obs_start:
            return

        # per-step confirmation: N consecutive low-std frames commit early,
        # without waiting for the window to end
        if ensemble_std is not None and ensemble_std < self.confidence_threshold:
            self.confidence_confirm_counter += 1
        else:
            self.confidence_confirm_counter = 0

        if steps_in_window > self.confidence_obs_end:
            self.confidence_phase = "recovering"
            self.confidence_recovery_step_count = 0
            self.get_logger().warning(
                f"[CONFIDENCE] Observation window ended. "
                f"counter={self.confidence_confirm_counter}/{self.confidence_confirm_frames}, "
                f"steps_in_window={steps_in_window}, entering recovery."
            )
            return

        # keep accumulating inside the window without judging
        # decide once when window end (step_in_window == confidence_obs_end
        if steps_in_window < self.confidence_obs_end:
            return

        # window just ended: one-shot decision using the window mean
        if np.isnan(self.ensemble_std_window_mean):
            # no valid data in the window: conservatively treat as failure
            self.confidence_phase = "recovering"
            self.confidence_recovery_step_count = 0
            self.get_logger().warning(
                f"[CONFIDENCE] No valid window data, entering recovery."
            )
            return

        self.get_logger().info(
            f"[CONFIDENCE] Window mean={self.ensemble_std_window_mean:.4f}, "
            f"threshold={self.confidence_threshold}, "
            f"steps_in_window={steps_in_window}"
        )

        if self.ensemble_std_window_mean < self.confidence_threshold:
            self.confidence_phase = "committed"
            self.get_logger().info(
                f"[CONFIDENCE] Committed! window_mean={self.ensemble_std_window_mean:.4f}"
            )
        else:
            self.confidence_phase = "recovering"
            self.confidence_recovery_step_count = 0
            self.get_logger().warning(
                f"[CONFIDENCE] Window mean too high "
                f"({self.ensemble_std_window_mean:.4f} >= {self.confidence_threshold}), "
                f"entering recovery."
            )

    # --------------------------------------------------
    # Timer callback
    # --------------------------------------------------
    def timer_callback(self):
        tic = time.time()

        try:
            with self.data_lock:
                current_mode = self.current_llm_mode

            mode_config = self.llm_modes.get(current_mode, self.llm_modes["all_gray"])
            mode_type = mode_config["type"]

            # all_gray / fixed_pose：skip ACT and publish the initial pose directly
            if mode_type == "fixed_pose":
                self.publish_fixed_action(mode_config["action"], current_mode)
                self.step += 1
                return

            qpos, curr_image, err = self.build_act_inputs()
            if err is not None:
                self.get_logger().warning(err)
                return

            # qpos logging: record the normalized qpos fed into ACT
            qpos_np_for_csv = qpos[0].detach().cpu().numpy()
            self.log_observations(qpos_np_for_csv)

            action, ensemble_std, ensemble_n = self.compute_action(qpos, curr_image)
            action = np.asarray(action, dtype=np.float32).reshape(-1)

            if self.action_dim is not None:
                zscore = np.abs((action - self.action_mean) / np.maximum(self.action_std, 1e-8))
                zscore_max = float(zscore.max())
                zscore_mean = float(zscore.mean())
            else:
                zscore_max = float("nan")
                zscore_mean = float("nan")

            self.get_logger().info(
                f"[METRIC] step={self.step}, mode={current_mode}, "
                f"ensemble_std={'nan' if ensemble_std is None else f'{ensemble_std:.4f}'}, "
                f"ensemble_n={ensemble_n}, "
                f"zscore_max={zscore_max:.4f}, zscore_mean={zscore_mean:.4f}"
            )

            self.steps_since_switch += 1

            self.update_ensemble_std_stats(ensemble_std)

            self.log_metrics(current_mode, ensemble_std, ensemble_n, zscore_max, zscore_mean)

            # --------------------------------------------------
            # Confidence assessment and action dispatch
            # --------------------------------------------------
            self.assess_confidence(ensemble_std)

            if self.confidence_phase == "recovering":
                self.confidence_recovery_step_count += 1
                self.publish_fixed_action(self.initial_action, "confidence_recovery")
                self.get_logger().info(
                    f"[CONFIDENCE] Recovering: {self.confidence_recovery_step_count}/{self.confidence_recovery_steps}"
                )
                if self.confidence_recovery_step_count >= self.confidence_recovery_steps:
                    self.confidence_phase = "observing"
                    self.confidence_confirm_counter = 0
                    self.confidence_recovery_step_count = 0
                    self.obs_window_start_step = self.steps_since_switch
                    self.ensemble_std_window_values = []
                    self.ensemble_std_window_mean = float("nan")
                    self.get_logger().info(
                        f"[CONFIDENCE] Recovery complete, re-entering observing. "
                        f"obs_window_start_step={self.obs_window_start_step}"
                    )
                self.step += 1
                return

            # observing or committed：publish the ACT action normally
            msg = Float32MultiArray()
            msg.data = action.tolist()
            self.action_pub.publish(msg)
            self.log_actions(action)

            elapsed_ms = (time.time() - tic) * 1000.0
            self.get_logger().info(
                f"step={self.step}, mode={current_mode}, "
                f"confidence_phase={self.confidence_phase}, "
                f"action_dim={len(action)}, "
                f"action_min={action.min():.6f}, action_max={action.max():.6f}, "
                f"inference_time={elapsed_ms:.2f} ms"
            )

            self.step += 1

        except Exception as e:
            self.get_logger().error(f"timer_callback failed: {e}")

    # --------------------------------------------------
    # Shutdown
    # --------------------------------------------------
    def destroy_node(self):
        self.get_logger().info("Shutting down ACT policy node...")

        try:
            if self.action_csv_file is not None:
                self.action_csv_file.close()
                self.get_logger().info("Action CSV file closed")
        except Exception as e:
            self.get_logger().warning(f"Failed to close action CSV file: {e}")

        try:
            if self.obs_csv_file is not None:
                self.obs_csv_file.close()
                self.get_logger().info("Observation CSV file closed")
        except Exception as e:
            self.get_logger().warning(f"Failed to close observation CSV file: {e}")

        try:
            if self.metric_csv_file is not None:
                self.metric_csv_file.close()
                self.get_logger().info("Metric CSV file closed")
        except Exception as e:
            self.get_logger().warning(f"Failed to close metric CSV file: {e}")

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = None
    executor = None
    try:
        node = ACTPolicyInferenceNode()

        # MultiThreadedExecutor do subscribers and the timer/inference do not block each other
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[ERROR] Node crashed: {e}")
    finally:
        # noinspection PyBroadException
        try:
            if executor is not None and node is not None:
                executor.remove_node(node)
        except Exception:
            pass  # best-effort cleanup; the ROS context may already be shut down

        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()