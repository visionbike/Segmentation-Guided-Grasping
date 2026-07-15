import time
from functools import partial
from operator import itemgetter

from . import task_logic
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from cv_bridge import CvBridge
from ultralytics import YOLO
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


class YoloSegNode(Node):

    def __init__(self):
        super().__init__("yolo_seg")

        self.declare_parameter("model", "")  # .pt file or OpenVINO IR dir (e.g. .../best_0613_openvino_model)
        self.declare_parameter("device", "intel:gpu")  # intel:gpu | intel:npu | intel:cpu | cpu
        self.declare_parameter("conf_threshold", 0.6)  # confidence threshold
        self.declare_parameter("input_width", 320)
        self.declare_parameter("input_height", 240)

        self.input_width = int(self.get_parameter("input_width").value)
        self.input_height = int(self.get_parameter("input_height").value)
        model_path = str(self.get_parameter("model").value)
        self.conf_threshold = float(self.get_parameter("conf_threshold").value)
        self.yolo_device = str(self.get_parameter("device").value)

        if not model_path:
            raise ValueError(
                "'model' parameter is empty - set it to the YOLO .pt file or "
                "OpenVINO IR dir (e.g. .../openvino_model)"
            )

        self.model = YOLO(model_path, task="segment")
        self.bridge = CvBridge()

        self.get_logger().info(f"Loaded YOLO model: {model_path} (device={self.yolo_device})")

        # BGR color definitions (this node publishes "bgr8")
        self.COLOR_BLACK = (0, 0, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_BLUE = (255, 0, 0)
        self.COLOR_YELLOW = (0, 255, 255)
        self.COLOR_CYAN = (255, 255, 0)
        self.COLOR_GRAY = (127, 127, 127)
        self.COLOR_PURPLE = (127, 0, 127)
        self.COLOR_WHITE = (189, 224, 255)

        # classes with fixed display colors
        # gripper is always purple
        self.fixed_class_color_map = {
            "gripper": self.COLOR_PURPLE,
        }

        self.default_color = self.COLOR_GRAY

        # LLM state comes from a single topic /llm_state (Float32MultiArray, 6-bit)
        #   format: [t0, t1, t2, d0, d1, d2]
        #     bits 0-2 = target      : red_cookies, green_tea, yellow_cookies
        #     bits 3-5 = destination : mid_basket, right_basket, human_hand
        #
        #   [1,0,0, 1,0,0] -> red_cookies_to_mid_basket
        #   [1,0,0, 0,1,0] -> red_cookies_to_right_basket
        #   [0,1,0, 1,0,0] -> green_tea_to_mid_basket
        #   [0,1,0, 0,1,0] -> green_tea_to_right_basket
        #   [0,0,1, 1,0,0] -> yellow_cookies_to_mid_basket
        #   [1,0,0, 0,0,1] -> red_cookies_to_human_hand
        #   [0,0,0, 0,0,0] -> all_gray
        self.TARGETS = ["red_cookies", "green_tea", "yellow_cookies"]
        self.DESTINATIONS = ["mid_basket", "right_basket", "human_hand"]

        self.valid_states = {
            "red_cookies_to_mid_basket",
            "red_cookies_to_right_basket",
            "green_tea_to_mid_basket",
            "green_tea_to_right_basket",
            "yellow_cookies_to_mid_basket",
            "red_cookies_to_human_hand",
            "green_tea_to_human_hand",
            "yellow_cookies_to_human_hand",
        }

        # before the first /llm_state arrives, lucky / tea stay black
        self.active_state_name = None
        self.latest_llm_state  = None

        # record object classes already inside a basket at the moment llm_state switches;
        # they stay painted in basket color until re-evaluated at the next llm_state switch
        self.completed_basket_slots = set()

        self.locked_target_objects = {}

        # ============================================================
        # shared across cameras: forced color override for non-target objects
        # when action_done is published, immediately record the color this cls should show
        # key: cls_name, value: "basket_color" | "black"
        # not cleared on state switch (persists across tasks); cleared only on all_gray
        # ============================================================
        self.non_target_color_override = {}


        # only these classes are checked for being inside a basket
        self.target_object_classes = {"lucky", "cheetos", "tea"}

        llm_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.llm_state_sub = self.create_subscription(
            Float32MultiArray,
            "/llm_state",
            self.llm_state_callback,
            llm_qos
        )

        # ============================================================
        # Action done publisher
        # when the target of the current active_state enters the basket, publish /action_done
        # ============================================================
        self.declare_parameter("enable_action_done", True)
        self.declare_parameter("action_done_topic", "/action_done")
        self.declare_parameter("done_overlap_ratio_thres", 0.05)
        self.declare_parameter("done_confirm_frames", 50)
        self.declare_parameter("done_cooldown_sec", 8.0)

        self.enable_action_done = bool(self.get_parameter("enable_action_done").value)
        self.action_done_topic = self.get_parameter("action_done_topic").value
        self.done_overlap_ratio_threshold = float(self.get_parameter("done_overlap_ratio_thres").value)
        self.done_confirm_frames = int(self.get_parameter("done_confirm_frames").value)
        self.done_cooldown_sec = float(self.get_parameter("done_cooldown_sec").value)

        self.action_done_pub = self.create_publisher(
            String,
            self.action_done_topic,
            10
        )

        # ============================================================
        # Water grasp detection → publish task_id=0 DONE to bridge
        # ============================================================
        self.declare_parameter("motor_feedback_topic", "/motor_angle_feedback_topic")
        self.declare_parameter("task_status_set_topic", "/task_status_set")
        self.declare_parameter("water_grasp_area_thres", 0.1)   # threshold for water mask area as fraction of the frame
        self.declare_parameter("water_grasp_gripper_thres", 0.5) # L_Gripper closed threshold (tune to actual values)
        self.declare_parameter("water_grasp_confirm_frames", 30)
        self.declare_parameter("water_grasp_fail_frames", 30)    # fail threshold: gripper closed but no water detected

        self.water_grasp_area_thres    = float(self.get_parameter("water_grasp_area_thres").value)
        self.water_grasp_gripper_thres = float(self.get_parameter("water_grasp_gripper_thres").value)
        self.water_grasp_confirm_frames = int(self.get_parameter("water_grasp_confirm_frames").value)
        self.water_grasp_fail_frames   = int(self.get_parameter("water_grasp_fail_frames").value)

        self.l_gripper_value           = 0.0
        self.water_grasp_counter       = 0
        self.water_grasp_fail_counter  = 0
        self.water_task_done_published  = False
        self.water_task_failed_published = False

        self.task_status_set_pub = self.create_publisher(
            Float32MultiArray,
            self.get_parameter("task_status_set_topic").value,
            10
        )

        self.create_subscription(
            Float32MultiArray,
            self.get_parameter("motor_feedback_topic").value,
            self.motor_feedback_callback,
            10
        )

        self.state_to_target_class = {
            "red_cookies_to_mid_basket": "lucky",
            "red_cookies_to_right_basket": "lucky",

            "green_tea_to_mid_basket": "tea",
            "green_tea_to_right_basket": "tea",

            "yellow_cookies_to_mid_basket": "cheetos",

            # human hand delivery mode
            "red_cookies_to_human_hand":    "lucky",
            "green_tea_to_human_hand":      "tea",
            "yellow_cookies_to_human_hand": "cheetos",
        }

        self.state_to_basket_side = {
            "red_cookies_to_mid_basket": "mid",
            "red_cookies_to_right_basket": "right",

            "green_tea_to_mid_basket": "mid",
            "green_tea_to_right_basket": "right",

            "yellow_cookies_to_mid_basket": "mid",

            # human hand mode shows no basket
            "red_cookies_to_human_hand":    None,
            "green_tea_to_human_hand":      None,
            "yellow_cookies_to_human_hand": None,

            # all_gray also shows no basket
            "all_gray": "all",
        }

        # reverse lookup: is the current state already completed
        # e.g.:
        #   green_tea_to_mid_basket   -> ("tea", "mid")
        #   green_tea_to_right_basket -> ("tea", "right")
        self.state_to_completed_slot = {}

        for state_name, target_cls in self.state_to_target_class.items():
            side = self.state_to_basket_side.get(state_name, None)

            if side in ["mid", "right"]:
                self.state_to_completed_slot[state_name] = (target_cls, side)

        # ============================================================
        # Basket relabel rules
        # YOLO only outputs "basket"; here basket instances are post-labeled
        # as mid_basket / right_basket depending on the camera view
        # ============================================================
        self.camera_basket_label_map = {
            # top: confirmed correct
            # smaller x = mid_basket, larger x = right_basket
            "top": {
                "left_label": "mid_basket",
                "right_label": "right_basket",
            },

            # right_wrist: confirmed correct
            # smaller x = mid_basket, larger x = right_basket
            "right_wrist": {
                "left_label": "mid_basket",
                "right_label": "right_basket",
            },

            # front: opposite of top
            # smaller x = right_basket, larger x = mid_basket
            "front": {
                "left_label": "right_basket",
                "right_label": "mid_basket",
            },

            # left_wrist: if two baskets are visible, assume same as top;
            # swap left_label / right_label if real tests show otherwise
            "left_wrist": {
                "left_label": "mid_basket",
                "right_label": "right_basket",
            },
        }

        # when only one basket is visible, do not guess from the current task;
        # left_wrist initially sees only mid_basket, so it is fixed to mid_basket
        self.camera_single_basket_label_map = {
            "left_wrist": "mid_basket",

            # other cameras label a single visible basket as unknown_basket
            # to avoid mistaking mid for right or vice versa
            "top": "unknown_basket",
            "right_wrist": "unknown_basket",
            "front": "unknown_basket",
        }

        # ============================================================
        # Object relabel rules
        # when two or more of the same class (lucky / cheetos / tea) appear,
        # relabel them as mid_object / right_object depending on the camera view.
        #
        # note: this is the reverse of the basket rule
        # ============================================================
        self.camera_object_label_map = {
            # top: basket rule is small x=mid, large x=right
            # objects reversed: small x=right_object, large x=mid_object
            "top": {
                "left_label": "right_object",
                "right_label": "mid_object",
            },

            # right_wrist: same as top
            # objects likewise reversed
            "right_wrist": {
                "left_label": "right_object",
                "right_label": "mid_object",
            },

            # front: basket rule is small x=right, large x=mid
            # objects reversed: small x=mid_object, large x=right_object
            "front": {
                "left_label": "mid_object",
                "right_label": "right_object",
            },

            # left_wrist: start with the object rule opposite to top;
            # swap left_label / right_label if real tests show otherwise
            "left_wrist": {
                "left_label": "right_object",
                "right_label": "mid_object",
            },
        }

        # when only one object is visible, keep it without guessing its position
        self.camera_single_object_label_map = {
            "top": "unknown_object",
            "right_wrist": "unknown_object",
            "front": "unknown_object",
            "left_wrist": "unknown_object",
        }

        # ============================================================
        # Object selection rules
        # when two or more of the same class appear, each camera deterministically picks larger/smaller x
        # ============================================================
        self.camera_object_select_rule = {
            "top": "max_x",  # top picks the larger-x object
            "front": "min_x",  # front picks the smaller-x object
            "left_wrist": "max_x",  # left wrist picks the larger-x object
            "right_wrist": "max_x",  # right wrist picks the larger-x object
        }

        # per-camera counter of consecutive successful frames
        self.done_confirm_counter = {
            "top": 0,
            "left_wrist": 0,
            "right_wrist": 0,
            "front": 0,
        }

        # avoid publishing action_done twice for the same state
        self.done_published_for_current_state = False
        self.last_done_publish_time = 0.0

        self.get_logger().info(f"[ACTION DONE] enable_action_done={self.enable_action_done}")
        self.get_logger().info(f"[ACTION DONE] topic={self.action_done_topic}")
        self.get_logger().info(f"[ACTION DONE] done_overlap_ratio_threshold={self.done_overlap_ratio_threshold}")
        self.get_logger().info(f"[ACTION DONE] done_confirm_frames={self.done_confirm_frames}")

        self.get_logger().info(
            "[LLM mask color] initial state: gripper=purple, basket=blue, others=gray"
        )

        self.topics = {
            "top": "/sync/top/image_raw",
            "left_wrist": "/sync/left_wrist/image_raw",
            "right_wrist": "/sync/right_wrist/image_raw",
            "front": "/sync/front/image_raw",
        }

        self.latest_msgs = {
            "top": None,
            "left_wrist": None,
            "right_wrist": None,
            "front": None,
        }

        self.latest_stamp_ns = {
            "top": -1,
            "left_wrist": -1,
            "right_wrist": -1,
            "front": -1,
        }

        self.last_processed_stamp_ns = {
            "top": -1,
            "left_wrist": -1,
            "right_wrist": -1,
            "front": -1,
        }

        self.pubs = {}

        for name, topic in self.topics.items():
            self.create_subscription(Image, topic, partial(self.store_msg, name=name), 1)

        for name in self.topics.keys():
            self.pubs[name] = self.create_publisher(Image, f"/{name}/YOLO_mask", 1)

        # fixed 10 Hz
        self.timer = self.create_timer(0.1, self.process_latest_frames_batch)

        self.frame_count = {
            "top": 0,
            "left_wrist": 0,
            "right_wrist": 0,
            "front": 0,
        }
        self.last_fps_print_time = time.time()


        self.camera_params = {
        # "top": {                      # D435
            #     "K": np.array([
            #         [306.935, 0.0, 160.757],
            #         [0.0, 307.02, 124.195],
            #         [0.0, 0.0, 1.0]
            #     ], dtype=np.float64),
            #     "D": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            #     # "K": None,
            #     # "D": None,
            # },
            "top": {
                "K": np.array([
                    [385.8769226074219, 0.0, 326.01617431640625],
                    [0.0, 385.3791809082031, 243.65151977539062],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64),
                "D": np.array([-0.05753061920404434, 0.06412331759929657, 0.00016869025421328843, 0.0007324286852963269, -0.020938973873853683], dtype=np.float64),
                # "K": None,
                # "D": None,
            },
            # "left_wrist": {             # D415
            #     "K": np.array([
            #         [304.0786, 0.0, 154.8443],
            #         [0.0, 303.5148, 120.2255],
            #         [0.0, 0.0, 1.0]
            #     ], dtype=np.float64),
            #     "D": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            #     # "K": None,
            #     # "D": None,
            # },
            "left_wrist": {             # D405
                "K": np.array([ 
                    [245.3480987548828, 0.0, 236.17245483398438],
                    [0.0, 244.7532958984375, 134.31321716308594],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64),
                "D": np.array([-0.05078468, 0.06116087, -0.00072102, -0.00013776, -0.02128844], dtype=np.float64),
                # "K": None,
                # "D": None,
            },
            # "right_wrist": {
            #     "K": np.array([
            #         [303.5599, 0.0, 158.1909],
            #         [0.0, 303.2278, 120.7818],
            #         [0.0, 0.0, 1.0]
            #     ], dtype=np.float64),
            #     "D": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            #     # "K": None,
            #     # "D": None,
            # },
            "right_wrist": {
                "K": np.array([
                    [244.27272033691406, 0.0, 240.11538696289062],
                    [0.0, 243.70840454101562, 134.84103393554688],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64),
                "D": np.array([-0.05039223 , 0.0607277 , -0.00067457 , 0.0013109 , -0.02094556], dtype=np.float64),
                # "K": None,
                # "D": None,
            },
            "front": {
                "K": np.array([
                    [314.9898, 0.0, 160.8542],
                    [0.0, 317.0161, 120.5791],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64),
                "D": np.array([-0.3685, 0.0989, 0.0, 0.0, 0.0], dtype=np.float64),
                # "K": None,
                # "D": None,
            },
        }

        # cache undistort maps to avoid recomputing every frame
        self.undistort_cache = {}

        self.get_logger().info("#############################################################")
        self.get_logger().info("YOLO_segmentation node is up!!!!")
        self.get_logger().info("Inference mode: latest frame only + batch inference @ 10 Hz")
        self.get_logger().info("#############################################################")

    def llm_state_callback(self, msg: Float32MultiArray):
        data = np.asarray(msg.data, dtype=np.float32).reshape(-1)

        expected_dim = len(self.TARGETS) + len(self.DESTINATIONS)  # 6
        if data.shape[0] != expected_dim:
            self.get_logger().warning(f"[/llm_state] dimension mismatch: got {data.shape[0]}, expected {expected_dim}")
            return

        target = data[:len(self.TARGETS)]
        dest   = data[len(self.TARGETS):]

        # ============================================================
        # special signal: all ones -> fully reset all history
        # different from all_gray [0,0,0,0,0,0]:
        #   all_gray   -> clear locked only, keep history (arm returns to initial position)
        #   full_reset -> clear all history (task sequence restarts from scratch)
        # ============================================================
        if np.min(data) >= 0.5:
            if self.active_state_name != "full_reset":
                self.active_state_name = "full_reset"
                self.completed_basket_slots.clear()
                self.locked_target_objects.clear()
                self.non_target_color_override.clear()
                self.done_published_for_current_state = False
                self.water_task_done_published = False
                self.water_task_failed_published = False
                self.water_grasp_counter = 0
                self.water_grasp_fail_counter = 0
                for cam_name in self.done_confirm_counter:
                    self.done_confirm_counter[cam_name] = 0
                self.get_logger().info(
                    "[FULL RESET] all history cleared: "
                    "completed_basket_slots, locked_target_objects, non_target_color_override"
                )
            return

        old_state_name = self.active_state_name
        self.latest_llm_state = data.copy()

        # [0,0,0,0,0,0] -> all_gray (clear locked only, keep history)
        if np.max(data) < 0.5:
            new_state_name = "all_gray"
        else:
            target_name = self.TARGETS[int(np.argmax(target))]
            dest_name   = self.DESTINATIONS[int(np.argmax(dest))]
            new_state_name = f"{target_name}_to_{dest_name}"

            if new_state_name not in self.valid_states:
                self.get_logger().warning(f"[LLM] Decoded invalid state: {new_state_name}, fallback to all_gray")
                new_state_name = "all_gray"

        if new_state_name != old_state_name:
            self.active_state_name = new_state_name

            if new_state_name == "all_gray":
                self.locked_target_objects.clear()
                self.get_logger().info(
                    "[TASK PAUSE] active_state=all_gray, "
                    "clear locked_target_objects only"
                )
            else:
                new_target_cls = self.state_to_target_class.get(new_state_name, None)
                if new_target_cls is not None and new_target_cls in self.non_target_color_override:
                    del self.non_target_color_override[new_target_cls]

            self.done_published_for_current_state = False
            self.water_task_done_published = False
            self.water_task_failed_published = False
            self.water_grasp_counter = 0
            self.water_grasp_fail_counter = 0

            for cam_name in self.done_confirm_counter:
                self.done_confirm_counter[cam_name] = 0

            self.get_logger().info(
                f"[LLM mask color] active_state changed: "
                f"{old_state_name} -> {new_state_name}, llm_state={data.tolist()}"
            )

    def store_msg(self, msg, name):
        self.latest_msgs[name] = msg
        self.latest_stamp_ns[name] = int(msg.header.stamp.sec) * 1000000000 + int(msg.header.stamp.nanosec)

    def get_undistort_maps(self, name, w, h):
        if name not in self.camera_params:
            return None

        K = self.camera_params[name]["K"]
        D = self.camera_params[name]["D"]

        if K is None or D is None:
            return None

        cache_key = (name, w, h)
        if cache_key in self.undistort_cache:
            return self.undistort_cache[cache_key]

        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0.0, (w, h))
        map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w, h), cv2.CV_16SC2)

        self.undistort_cache[cache_key] = (map1, map2, new_K, roi)
        # self.get_logger().info(f"[{name}] undistort map created for size {w}x{h}")
        return self.undistort_cache[cache_key]

    def undistort_image(self, name: str, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        maps = self.get_undistort_maps(name, w, h)

        if maps is None:
            return img

        map1, map2, _, _ = maps  # new_K / roi unused here (kept in the cache entry)
        undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        return undist

    def resize_with_padding(self, img):
        h, w = img.shape[:2]

        if w == self.input_width and h == self.input_height:
            return img

        # scale keeping aspect ratio
        scale = min(self.input_width / w, self.input_height / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )

        # create black canvas
        padded = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)

        # paste centered
        x_offset = (self.input_width - new_w) // 2
        y_offset = (self.input_height - new_h) // 2

        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        # self.get_logger().warn(
        #     f"Input image size is {w}x{h}, resized with padding to "
        #     f"{self.input_width}x{self.input_height} "
        #     f"(scaled={new_w}x{new_h}, offset=({x_offset},{y_offset}))"
        # )

        return padded

    def process_latest_frames_batch(self):
        t0 = time.time()

        batch_names = []
        batch_msgs = []
        batch_imgs = []

        # collect all cameras that have a new frame
        for name in self.topics.keys():
            msg = self.latest_msgs[name]
            if msg is None:
                continue

            stamp_ns = self.latest_stamp_ns[name]
            if stamp_ns == self.last_processed_stamp_ns[name]:
                continue

            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # undistort before feeding YOLO
            img_undist = self.undistort_image(name, img)

            # make sure the YOLO input is exactly 320x240
            img_ready = self.resize_with_padding(img_undist)

            batch_names.append(name)
            batch_msgs.append(msg)
            batch_imgs.append(img_ready)

        if len(batch_imgs) == 0:
            return

        # t1 = time.time()

        # single batched inference
        results = self.model(
            batch_imgs,
            imgsz=320,
            verbose=False,
            device=self.yolo_device,
            conf=self.conf_threshold   # confidence filtering at inference time
        )

        # t2 = time.time()

        for name, msg, img_ready, result in zip(batch_names, batch_msgs, batch_imgs, results):
            self.publish_color_mask(name, msg, img_ready, result)
            self.last_processed_stamp_ns[name] = self.latest_stamp_ns[name]
            self.frame_count[name] += 1

        t3 = time.time()

        # print output rate once per second
        now = time.time()
        if now - self.last_fps_print_time >= 1.0:
            # fps_text = ", ".join(
            #     [f"{name}: {self.frame_count[name]:.1f} Hz" for name in self.topics.keys()]
            # )
            # self.get_logger().info(f"[YOLO output rate] {fps_text}")
            for name in self.frame_count:
                self.frame_count[name] = 0
            self.last_fps_print_time = now

        # prep_ms = (t1 - t0) * 1000.0
        # infer_ms = (t2 - t1) * 1000.0
        # post_ms = (t3 - t2) * 1000.0
        total_ms = (t3 - t0) * 1000.0

        # self.get_logger().info(
        #     f"[BATCH {len(batch_imgs)} imgs] prep={prep_ms:.1f} ms, "
        #     f"infer={infer_ms:.1f} ms, post+pub={post_ms:.1f} ms, total={total_ms:.1f} ms"
        # )

        if total_ms > 100:
            self.get_logger().warning(f"process_latest_frames_batch took {total_ms:.1f} ms")

    def get_current_target_class(self):
        """
        Return the target object class to grasp for the current active_state.
        For example:
            red_cookies_to_mid_basket      -> lucky
            red_cookies_to_right_basket    -> lucky
            green_tea_to_mid_basket        -> tea
            green_tea_to_right_basket      -> tea
            yellow_cookies_to_mid_basket   -> cheetos
            red_cookies_to_human_hand      -> lucky
        """
        if self.active_state_name is None:
            return None

        return self.state_to_target_class.get(self.active_state_name, None)
    
    def is_current_state_completed(self):
        """
        Check whether the current active_state is already completed.

        For example:
            active_state = green_tea_to_mid_basket
            completed_basket_slots contains ("tea", "mid")
            -> True

            active_state = green_tea_to_right_basket
            completed_basket_slots has ("tea", "mid") but not ("tea", "right")
            -> False
        """

        if self.active_state_name is None:
            return False

        completed_slot = self.state_to_completed_slot.get(
            self.active_state_name,
            None
        )

        if completed_slot is None:
            return False

        return completed_slot in self.completed_basket_slots

    def relabel_basket_instances(self, basket_instances, camera_name):
        """Delegates to task_logic with this node's camera label rules."""
        return task_logic.relabel_basket_instances(
            basket_instances, camera_name,
            self.camera_basket_label_map, self.camera_single_basket_label_map
        )
    
    def relabel_object_instances(self, object_instances, camera_name):
        """Delegates to task_logic with this node's camera label rules."""
        return task_logic.relabel_object_instances(
            object_instances, camera_name,
            self.camera_object_label_map, self.camera_single_object_label_map
        )
    
    def select_object_instance_by_camera_x(self, instances, camera_name):
        """Delegates to task_logic with this node's per-camera selection rule."""
        return task_logic.select_object_instance_by_camera_x(
            instances, camera_name, self.camera_object_select_rule
        )

    def select_or_track_locked_object_instance(self, instances, camera_name, cls_name):
        """
        Within one task, duplicated objects are selected only once by the camera rule.

        First time:
            top / left_wrist / right_wrist -> pick larger x
            front -> pick smaller x

        Afterwards:
            no longer re-evaluated with max_x / min_x;
            instead pick the instance closest to the last locked centroid,
            so selection does not jump to another same-class object due to motion, basket entry, or YOLO mask jitter.

        On task end or state change, llm_state_callback() clears self.locked_target_objects.
        """

        if len(instances) == 0:
            return []

        key = (camera_name, cls_name)

        # =====================================================
        # Case 1: this camera + cls is not locked yet in this task.
        # Only the first time is selected via camera_object_select_rule.
        # =====================================================
        if key not in self.locked_target_objects:
            if len(instances) == 1:
                selected = instances[0]
            else:
                rule = self.camera_object_select_rule.get(camera_name, "max_x")

                if rule == "min_x":
                    selected = min(instances, key=itemgetter("cx"))
                else:
                    selected = max(instances, key=itemgetter("cx"))

                # self.get_logger().info(
                #     f"[OBJECT LOCK INIT] camera={camera_name}, "
                #     f"cls={cls_name}, rule={rule}, "
                #     f"all_cx={[round(item['cx'], 1) for item in instances]}, "
                #     f"locked_cx={selected['cx']:.1f}, "
                #     f"locked_cy={selected['cy']:.1f}, "
                #     f"state={self.active_state_name}"
                # )

            self.locked_target_objects[key] = {
                "cx": selected["cx"],
                "cy": selected["cy"],
                "state": self.active_state_name,
            }

            return [selected]

        # =====================================================
        # Case 2: already locked.
        # No longer re-select by max_x / min_x; track the instance closest to the last frame's locked centroid.
        # =====================================================
        locked = self.locked_target_objects[key]
        last_cx = locked["cx"]
        last_cy = locked["cy"]

        selected = min(instances, key=task_logic.sq_dist_to(last_cx, last_cy))

        # update the locked centroid so it follows the same object
        self.locked_target_objects[key] = {
            "cx": selected["cx"],
            "cy": selected["cy"],
            "state": self.active_state_name,
        }

        # self.get_logger().info(
        #     f"[OBJECT LOCK TRACK] camera={camera_name}, "
        #     f"cls={cls_name}, "
        #     f"prev=({last_cx:.1f}, {last_cy:.1f}), "
        #     f"new=({selected['cx']:.1f}, {selected['cy']:.1f}), "
        #     f"all_cx={[round(item['cx'], 1) for item in instances]}"
        # )

        return [selected]

    @staticmethod
    def is_object_inside_basket(obj_mask, basket_cover_mask, ratio_thres=0.30):
        """Delegates to task_logic (pure overlap test)."""
        return task_logic.is_object_inside_basket(obj_mask, basket_cover_mask, ratio_thres)
    
    def filter_object_masks_by_current_task(self, object_masks, camera_name, basket_cover_mask):
        """
        When two or more of the same class (lucky / cheetos / tea) appear,
        select once by camera rule only at the start of the task.

        Key logic:
        1. If this camera + class is already locked:
        - do not exclude objects inside the basket anymore
        - keep tracking the originally locked instance
        - reset only after /action_done is published

        2. If not locked yet:
        - first exclude instances already inside the basket
        - to avoid selecting an already-completed object at task start
        """

        # all_gray / initial state: no object filtering
        if self.active_state_name is None or self.active_state_name in ("all_gray", "full_reset"):
            return object_masks

        filtered_objects = []

        grouped = {}
        for obj in object_masks:
            cls_name = obj["cls_name"]
            grouped.setdefault(cls_name, []).append(obj)

        for cls_name, instances in grouped.items():

            # keep all other classes (gripper / hand, etc.) as-is
            if cls_name not in self.target_object_classes:
                filtered_objects.extend(instances)
                continue

            key = (camera_name, cls_name)

            # =====================================================
            # non-target with an override record (was placed into a basket):
            # keep all instances; stage 4 paints them black or blue uniformly.
            # no selection or lock/track needed.
            # =====================================================
            if cls_name != self.get_current_target_class() and cls_name in self.non_target_color_override:
                filtered_objects.extend(instances)
                continue

            # =====================================================
            # Case 0:
            # if the slot for the current active_state is already completed,
            # do not let same-class targets outside the basket light up again.
            #
            # e.g.:
            #   ("tea", "mid") completed, but current state is green_tea_to_right_basket
            #   -> does not block the right-basket task; another tea can still be selected
            #
            #   ("tea", "right") completed and current state is green_tea_to_right_basket
            #   -> block other teas outside the basket so another tea does not light up
            # =====================================================
            current_target_cls = self.get_current_target_class()

            if self.is_current_state_completed() and cls_name == current_target_cls:
                in_basket_instances = []

                for obj in instances:
                    inside_basket = self.is_object_inside_basket(
                        obj_mask=obj["mask"],
                        basket_cover_mask=basket_cover_mask,
                        ratio_thres=0.30
                    )

                    if inside_basket:
                        in_basket_instances.append(obj)

                filtered_objects.extend(in_basket_instances)

                # self.get_logger().info(
                #     f"[OBJECT FILTER CURRENT SLOT COMPLETED] camera={camera_name}, "
                #     f"state={self.active_state_name}, "
                #     f"cls={cls_name}, "
                #     f"completed_slots={self.completed_basket_slots}, "
                #     f"total={len(instances)}, "
                #     f"in_basket={len(in_basket_instances)}, "
                #     f"skip outside target objects"
                # )

                continue

            # =====================================================
            # Case 1:
            # this camera + class is already locked.
            # Never exclude instances inside the basket here, otherwise the object
            # is excluded as soon as it enters the basket and tracking jumps to another same-class object.
            # =====================================================
            if key in self.locked_target_objects:
                candidate_instances = instances

                # self.get_logger().info(
                #     f"[OBJECT FILTER LOCKED] camera={camera_name}, "
                #     f"cls={cls_name}, "
                #     f"keep tracking locked object, "
                #     f"candidate={len(candidate_instances)}"
                # )

            # =====================================================
            # Case 2:
            # not locked yet: this is the first selection at task start.
            # Only now exclude instances already inside the basket.
            # =====================================================
            else:
                outside_basket_instances = []

                for obj in instances:
                    inside_basket = self.is_object_inside_basket(
                        obj_mask=obj["mask"],
                        basket_cover_mask=basket_cover_mask,
                        ratio_thres=0.30
                    )

                    if not inside_basket:
                        outside_basket_instances.append(obj)

                # if objects exist outside the basket, select only among those;
                # if all are inside the basket, fall back to all instances so something is still shown
                if len(outside_basket_instances) > 0:
                    candidate_instances = outside_basket_instances
                else:
                    candidate_instances = instances

                # self.get_logger().info(
                #     f"[OBJECT FILTER INIT] camera={camera_name}, "
                #     f"cls={cls_name}, "
                #     f"total={len(instances)}, "
                #     f"outside_basket={len(outside_basket_instances)}, "
                #     f"candidate={len(candidate_instances)}"
                # )

            selected = self.select_or_track_locked_object_instance(
                instances=candidate_instances,
                camera_name=camera_name,
                cls_name=cls_name
            )

            filtered_objects.extend(selected)

        return filtered_objects
    
    @staticmethod
    def select_basket_mask_by_side(labeled_baskets, side, h, w):
        """Delegates to task_logic (pure mask selection)."""
        return task_logic.select_basket_mask_by_side(labeled_baskets, side, h, w)
        
    @staticmethod
    def select_basket_mask_by_sides(labeled_baskets, sides, h, w):
        """Delegates to task_logic (pure mask selection)."""
        return task_logic.select_basket_mask_by_sides(labeled_baskets, sides, h, w)
    
    @staticmethod
    def make_basket_cover_mask(basket_union_mask):
        """Delegates to task_logic (pure convex-hull cover mask)."""
        return task_logic.make_basket_cover_mask(basket_union_mask)

    def get_current_basket_side(self):
        if self.active_state_name is None:
            return "all"

        # full_reset: show no basket
        if self.active_state_name == "full_reset":
            return None

        return self.state_to_basket_side.get(self.active_state_name, "all")

    def get_dynamic_mask_color(self, cls_name: str):
        """
        Decide the mask color from the current LLM state.
        This node publishes bgr8, so colors are BGR.
        """

        # basket color:
        # whether the basket is displayed is NOT decided here;
        # this only decides what color a selected basket is painted.
        if cls_name == "basket":
            return self.COLOR_BLUE

        # gripper fixed color
        if cls_name in self.fixed_class_color_map:
            return self.fixed_class_color_map[cls_name]

        if self.active_state_name is None:
            return self.default_color
        
        # all_gray: everything except fixed classes is gray
        if self.active_state_name == "all_gray":
            return self.COLOR_GRAY

        # red cookies -> middle basket
        if self.active_state_name == "red_cookies_to_mid_basket":
            # if cls_name == "lucky":
            #     return self.COLOR_RED
            # elif cls_name in ["cheetos", "tea"]:
            #     return self.COLOR_GRAY
            if cls_name == "water":
                return self.COLOR_CYAN
            elif cls_name in ["cheetos", "tea"]:
                return self.COLOR_GRAY

        # red cookies -> right basket
        elif self.active_state_name == "red_cookies_to_right_basket":
            if cls_name == "lucky":
                return self.COLOR_RED
            elif cls_name in ["cheetos", "tea"]:
                return self.COLOR_GRAY

        # green tea -> middle basket
        elif self.active_state_name == "green_tea_to_mid_basket":
            if cls_name == "tea":
                return self.COLOR_GREEN
            elif cls_name in ["lucky", "cheetos"]:
                return self.COLOR_GRAY

        # green tea -> right basket
        elif self.active_state_name == "green_tea_to_right_basket":
            if cls_name == "tea":
                return self.COLOR_GREEN
            elif cls_name in ["lucky", "cheetos"]:
                return self.COLOR_GRAY

        # yellow cookies -> middle basket
        elif self.active_state_name == "yellow_cookies_to_mid_basket":
            if cls_name == "cheetos":
                return self.COLOR_YELLOW
            elif cls_name in ["lucky", "tea"]:
                return self.COLOR_GRAY

        # red cookies -> human hand
        elif self.active_state_name == "red_cookies_to_human_hand":
            if cls_name == "hand":
                return self.COLOR_WHITE
            elif cls_name == "lucky":
                return self.COLOR_RED
            elif cls_name in ["cheetos", "tea"]:
                return self.COLOR_GRAY
            elif cls_name == "basket":
                return self.COLOR_BLACK

        # green tea -> human hand
        elif self.active_state_name == "green_tea_to_human_hand":
            if cls_name == "hand":
                return self.COLOR_WHITE
            elif cls_name == "tea":
                return self.COLOR_GREEN
            elif cls_name in ["lucky", "cheetos"]:
                return self.COLOR_GRAY
            elif cls_name == "basket":
                return self.COLOR_BLACK

        # yellow cookies -> human hand
        elif self.active_state_name == "yellow_cookies_to_human_hand":
            if cls_name == "hand":
                return self.COLOR_WHITE
            elif cls_name == "cheetos":
                return self.COLOR_YELLOW
            elif cls_name in ["lucky", "tea"]:
                return self.COLOR_GRAY
            elif cls_name == "basket":
                return self.COLOR_BLACK

        return self.default_color
    
    def check_and_publish_action_done(self, name, object_masks, basket_cover_mask):
        """
        Check whether the target of the current active_state has entered the basket.
        If the overlap ratio holds for done_confirm_frames consecutive frames, publish /action_done.

        For example:
            active_state_name = lucky_active
            target_class = lucky

        If the lucky mask overlaps basket_cover_mask enough:
            publish /action_done: lucky_active
        """

        if not self.enable_action_done:
            return

        if self.done_published_for_current_state:
            return

        if self.active_state_name is None:
            return

        if self.active_state_name in ("all_gray", "full_reset"):
            return

        if self.active_state_name not in self.state_to_target_class:
            return

        if not np.any(basket_cover_mask):
            self.done_confirm_counter[name] = 0
            return

        target_cls = self.state_to_target_class[self.active_state_name]

        max_overlap_ratio = 0.0
        target_detected = False

        for obj in object_masks:
            cls_name = obj["cls_name"]
            obj_mask = obj["mask"]

            if cls_name != target_cls:
                continue

            obj_area = np.count_nonzero(obj_mask)
            if obj_area == 0:
                continue

            target_detected = True

            overlap_area = np.count_nonzero(obj_mask & basket_cover_mask)
            overlap_ratio = overlap_area / float(obj_area)

            if overlap_ratio > max_overlap_ratio:
                max_overlap_ratio = overlap_ratio

        if not target_detected:
            self.done_confirm_counter[name] = 0
            return

        if max_overlap_ratio >= self.done_overlap_ratio_threshold:
            self.done_confirm_counter[name] += 1

            # self.get_logger().info(
            #     f"[ACTION DONE CHECK][{name}] "
            #     f"state={self.active_state_name}, "
            #     f"target={target_cls}, "
            #     f"overlap_ratio={max_overlap_ratio:.2f}, "
            #     f"counter={self.done_confirm_counter[name]}/{self.done_confirm_frames}"
            # )
        else:
            self.done_confirm_counter[name] = 0

        if self.done_confirm_counter[name] < self.done_confirm_frames:
            return

        now = time.time()

        if now - self.last_done_publish_time < self.done_cooldown_sec:
            return

        done_msg = String()
        done_msg.data = self.active_state_name
        self.action_done_pub.publish(done_msg)

        self.done_published_for_current_state = True
        self.last_done_publish_time = now

        # record which target class + basket side this task completed
        done_side = self.get_current_basket_side()

        if done_side in ["mid", "right"]:
            self.completed_basket_slots.add((target_cls, done_side))

            # self.get_logger().info(
            #     f"[COMPLETED BASKET SLOT] "
            #     f"target={target_cls}, side={done_side}, "
            #     f"completed_slots={self.completed_basket_slots}"
            # )

        # ============================================================
        # right after action_done, record the color override for this cls
        # so all cameras show the correct color from the next frame on
        # ============================================================
        self.non_target_color_override[target_cls] = {
            "side": done_side,
        }

        # self.get_logger().info(
        #     f"[COLOR OVERRIDE SET] cls={target_cls}, side={done_side}"
        # )

        # only after action_done may objects be re-selected next time
        self.locked_target_objects.clear()

        # self.get_logger().info(
        #     f"[ACTION DONE PUBLISH] topic={self.action_done_topic}, "
        #     f"data={done_msg.data}, "
        #     f"camera={name}, "
        #     f"target={target_cls}, "
        #     f"overlap_ratio={max_overlap_ratio:.2f}"
        # )

    def motor_feedback_callback(self, msg: Float32MultiArray):
        data = list(msg.data)
        # format: [R1..R7, L1..L7, R_gripper, L_gripper, Neck]  (17 values)
        if len(data) < 16:
            return
        self.l_gripper_value = float(data[15])

    def check_and_publish_water_grasped(self, name, object_masks, total_pixels):
        if name != "left_wrist":
            return

        if self.water_task_done_published or self.water_task_failed_published:
            return

        if self.active_state_name != "red_cookies_to_mid_basket":
            return

        # L_Gripper not closed -> reset both counters
        if self.l_gripper_value < self.water_grasp_gripper_thres:
            self.water_grasp_counter = 0
            self.water_grasp_fail_counter = 0
            return

        # is the water mask area fraction in left_wrist above the threshold
        water_found = False
        for obj in object_masks:
            if obj["cls_name"] != "water":
                continue
            area = int(np.count_nonzero(obj["mask"]))
            ratio = area / float(total_pixels)
            if ratio >= self.water_grasp_area_thres:
                water_found = True
                break

        if water_found:
            # success path: water detected
            self.water_grasp_fail_counter = 0
            self.water_grasp_counter += 1
            self.get_logger().info(
                f"[WATER GRASP] counter={self.water_grasp_counter}/{self.water_grasp_confirm_frames}, "
                f"L_gripper={self.l_gripper_value:.3f}"
            )

            if self.water_grasp_counter < self.water_grasp_confirm_frames:
                return

            status_msg = Float32MultiArray()
            status_msg.data = [0.0, 1.0]  # task_id=0, done=1
            self.task_status_set_pub.publish(status_msg)
            self.water_task_done_published = True
            self.get_logger().info(
                "[WATER GRASP] Task done! Published [0.0, 1.0] to /task_status_set"
            )
        else:
            # failure path: gripper closed but no water detected
            self.water_grasp_counter = 0
            self.water_grasp_fail_counter += 1
            self.get_logger().info(
                f"[WATER GRASP FAIL] fail_counter={self.water_grasp_fail_counter}/{self.water_grasp_fail_frames}, "
                f"L_gripper={self.l_gripper_value:.3f}"
            )

            if self.water_grasp_fail_counter < self.water_grasp_fail_frames:
                return

            status_msg = Float32MultiArray()
            status_msg.data = [0.0, -1.0]  # task_id=0, failed=-1
            self.task_status_set_pub.publish(status_msg)
            self.water_task_failed_published = True
            self.get_logger().info(
                "[WATER GRASP FAIL] Task failed! Published [0.0, -1.0] to /task_status_set"
            )

    def publish_color_mask(self, name, msg, img, result):
        h, w = img.shape[:2]
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        basket_instances = []
        # basket_union_mask = np.zeros((h, w), dtype=bool)
        object_masks = []

        # =========================================================
        # Stage 1: collect basket masks and other object masks first
        # do not paint colors here
        # =========================================================
        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            names = result.names

            for i in range(len(masks)):
                cls_id = classes[i]
                cls_name = names[cls_id]

                obj_mask = np.asarray(cv2.resize(
                    masks[i].astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )) > 0

                if cls_name == "basket":
                    ys, xs = np.where(obj_mask)

                    if len(xs) > 0:
                        cx = float(np.mean(xs))
                    else:
                        cx = 0.0

                    basket_instances.append({
                        "mask": obj_mask,
                        "cx": cx,
                    })
                else:
                    ys, xs = np.where(obj_mask)

                    if len(xs) > 0:
                        cx = float(np.mean(xs))
                        cy = float(np.mean(ys))
                    else:
                        cx = 0.0
                        cy = 0.0

                    object_masks.append({
                        "cls_name": cls_name,
                        "mask": obj_mask,
                        "cx": cx,
                        "cy": cy,
                    })

        # =========================================================
        # Stage 2: build basket_cover_mask
        # used only to check whether objects are inside the basket,
        # not painted onto color_mask directly
        # =========================================================
        # post-label YOLO basket instances as mid_basket / right_basket
        labeled_baskets = self.relabel_basket_instances(
            basket_instances=basket_instances,
            camera_name=name
        )

        current_basket_side = self.get_current_basket_side()

        # basket targeted by the current task
        current_basket_mask = self.select_basket_mask_by_side(
            labeled_baskets=labeled_baskets,
            side=current_basket_side,
            h=h,
            w=w
        )

        # =========================================================
        # completed basket sides
        # note:
        # no longer filtered by current_target_cls, because objects already placed
        # in a basket must keep the basket color even while other tasks run.
        # =========================================================
        all_completed_sides = set()

        for completed_cls_name, completed_side in self.completed_basket_slots:
            if completed_side is not None:
                all_completed_sides.add(completed_side)

        completed_basket_mask = self.select_basket_mask_by_sides(
            labeled_baskets=labeled_baskets,
            sides=all_completed_sides,
            h=h,
            w=w
        )

        # =========================================================
        # only the current task's basket is displayed;
        # do not draw completed_basket_mask too, otherwise the mid basket
        # would also appear when switching to the right-basket task.
        # =========================================================
        basket_union_mask = current_basket_mask

        # current-task basket cover: used only for the action_done check
        basket_cover_mask = self.make_basket_cover_mask(current_basket_mask)

        if basket_cover_mask is None:
            basket_cover_mask = np.zeros((h, w), dtype=bool)

        # completed basket cover: excludes completed objects so the next task does not select them
        completed_basket_cover_mask = self.make_basket_cover_mask(completed_basket_mask)

        if completed_basket_cover_mask is None:
            completed_basket_cover_mask = np.zeros((h, w), dtype=bool)

        # =========================================================
        # build a cover mask for each completed basket side
        # to determine whether a completed object is in mid or right
        #
        # note:
        # this block must stay outside the "if completed_basket_cover_mask is None",
        # otherwise completed_basket_cover_by_side is not built when a completed basket exists.
        # =========================================================
        # NOTE: currently unused (never read) - kept for reference
        # completed_basket_cover_by_side = {}
        #
        # for side in ["mid", "right"]:
        #
        #     # if this side is not completed yet, use an all-black mask;
        #     # later checks also skip it because side not in all_completed_sides
        #     if side not in all_completed_sides:
        #         completed_basket_cover_by_side[side] = np.zeros((h, w), dtype=bool)
        #         continue
        #
        #     side_mask = self.select_basket_mask_by_sides(
        #         labeled_baskets=labeled_baskets,
        #         sides={side},
        #         h=h,
        #         w=w
        #     )
        #
        #     side_cover = self.make_basket_cover_mask(side_mask)
        #
        #     if side_cover is None:
        #         side_cover = np.zeros((h, w), dtype=bool)
        #
        #     completed_basket_cover_by_side[side] = side_cover

        # when filtering objects, exclude those inside the current basket AND inside completed baskets
        exclude_basket_cover_mask = basket_cover_mask | completed_basket_cover_mask

        basket_color = self.get_dynamic_mask_color("basket")

        # =========================================================
        # for duplicated target objects, keep only the one the current task needs
        # e.g. when two lucky appear, keep only mid_object or right_object
        # note: this step must run before action_done and the basket check
        # =========================================================
        object_masks = self.filter_object_masks_by_current_task(
            object_masks=object_masks,
            camera_name=name,
            basket_cover_mask=exclude_basket_cover_mask
        )

        # =========================================================
        # every frame, check whether the current active target has entered the basket;
        # if so, publish /action_done to the LLM sequence node
        # =========================================================
        self.check_and_publish_action_done(
            name=name,
            object_masks=object_masks,
            basket_cover_mask=basket_cover_mask
        )

        self.check_and_publish_water_grasped(
            name=name,
            object_masks=object_masks,
            total_pixels=h * w
        )

        # =========================================================
        # Stage 4: paint objects
        #
        # Rules:
        #   1. all_gray -> gripper purple, others gray
        #   2. target object -> LLM state color (red/green/yellow)
        #   3. non-target target_object_classes:
        #      - judged placed into any basket (recorded in completed_basket_slots)
        #        and currently inside that basket's cover:
        #          -> that basket is the current task basket -> blue
        #          -> that basket is not the current task basket -> black
        #      - not inside any completed basket, but completed_basket_slots has a record for the class
        #        -> black (was placed before but this camera cannot see the basket)
        #      - no completed record at all -> gray
        #   4. fixed classes like gripper / hand -> fixed colors
        # =========================================================
        for obj in object_masks:
            cls_name = obj["cls_name"]
            obj_mask = obj["mask"]

            current_target_cls = self.get_current_target_class()

            # =====================================================
            # all_gray state
            # =====================================================
            if self.active_state_name in ("all_gray", "full_reset"):
                if cls_name in self.fixed_class_color_map:
                    color_mask[obj_mask] = self.fixed_class_color_map[cls_name]
                elif cls_name in self.non_target_color_override:
                    color_mask[obj_mask] = self.COLOR_BLACK
                else:
                    color_mask[obj_mask] = self.COLOR_GRAY
                continue

            # =====================================================
            # initial state (no llm_state received yet)
            # =====================================================
            if self.active_state_name is None:
                color = self.get_dynamic_mask_color(cls_name)
                color_mask[obj_mask] = color
                continue

            # =====================================================
            # fixed classes like gripper / hand -> fixed colors, unaffected by any logic
            # =====================================================
            if cls_name not in self.target_object_classes:
                color = self.get_dynamic_mask_color(cls_name)
                color_mask[obj_mask] = color
                continue

            # =====================================================
            # target object:
            # paint the whole object in the LLM state color first,
            # then overwrite the basket-overlapping part with the basket color,
            # so the overlap shows blue correctly once the object is in the basket.
            # =====================================================
            if cls_name == current_target_cls:
                if self.done_published_for_current_state:
                    color_mask[obj_mask] = self.COLOR_BLACK
                    continue

                # check whether this object is already in another completed basket;
                # e.g. during green_tea_to_mid_basket, the tea in the right basket must not light up
                if cls_name in self.non_target_color_override:
                    override = self.non_target_color_override[cls_name]
                    override_side = override["side"]
                    if override_side != current_basket_side:
                        # this object is in another completed basket: paint black
                        color_mask[obj_mask] = self.COLOR_BLACK
                        continue
                    else:
                        # in the current basket: paint blue (overwritten by stage 5)
                        color_mask[obj_mask] = self.COLOR_BLACK
                        continue

                # not completed and no override -> normal LLM state color
                color = self.get_dynamic_mask_color(cls_name)
                color_mask[obj_mask] = color
                continue

            # =====================================================
            # non-target target_object_classes (e.g. tea/cheetos during a lucky task):
            #
            # look up completed_basket_slots for a record of this cls_name;
            # a record means this object was placed into some basket.
            #
            # then check whether it is currently inside that basket's cover:
            #   inside the current basket -> blue
            #   inside a non-current basket -> black
            #   not inside any basket (but has a record) -> black
            #
            # no record at all -> gray
            # =====================================================
            # obj_area = np.count_nonzero(obj_mask)

            # =====================================================
            # check non_target_color_override first (takes effect right after action_done)
            # so all cameras unify the color on the next frame without per-frame overlap computation
            # =====================================================
            if cls_name in self.non_target_color_override:
                override = self.non_target_color_override[cls_name]
                override_side = override["side"]
                if override_side == current_basket_side:
                    color_mask[obj_mask] = basket_color
                else:
                    color_mask[obj_mask] = self.COLOR_BLACK
            else:
                # check completed_basket_slots for a record of this cls
                completed_sides_for_cls = {
                    side for (c, side) in self.completed_basket_slots
                    if c == cls_name and side in ["mid", "right"]
                }
                if len(completed_sides_for_cls) > 0:
                    # was placed before but the override was cleared (it is now the target's class) -> black
                    color_mask[obj_mask] = self.COLOR_BLACK
                else:
                    color_mask[obj_mask] = self.COLOR_GRAY

        # =========================================================
        # Stage 5: paint the basket itself
        # draw the basket outline with basket_union_mask,
        # then paint every black region inside basket_cover_mask (filled version) blue,
        # so the basket area shows blue correctly when a completed target turns black.
        # =========================================================
        # =========================================================
        # Stage 5: paint the basket itself
        # =========================================================
        if np.any(basket_union_mask):
            empty_region = np.all(color_mask == self.COLOR_BLACK, axis=2)

            # basket outline itself
            draw_basket_mask = basket_union_mask & empty_region
            color_mask[draw_basket_mask] = basket_color

            # basket interior fill
            if np.any(basket_cover_mask):
                inner_region = basket_cover_mask & empty_region
                color_mask[inner_region] = basket_color

        # =========================================================
        # publish: must publish every frame, basket or not
        # =========================================================
        mask_msg = self.bridge.cv2_to_imgmsg(color_mask, "bgr8")
        mask_msg.header = msg.header
        self.pubs[name].publish(mask_msg)

def main():
    rclpy.init()
    node = YoloSegNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
