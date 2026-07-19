# Segmentation-Guided Grasping (branch: intel-deployment)

ACT policy driven by task-conditioned YOLO segmentation masks,
deployed with OpenVINO on an Intel AI PC (ROS 2, ament_python).

## Pipeline

```
cameras (3x RealSense: top, left_wrist, right_wrist)
    └─> obs_sync                  sync images + encoder state
            /sync/{cam}/image_raw, /sync/qpos
    └─> yolo_seg                  task-conditioned segmentation (OpenVINO, Intel GPU/NPU)
            /{cam}/YOLO_mask, /action_done
    └─> act_policy                ACT inference (openvino | onnx | torch backend)
            /motor_action_angle_topic
    └─> packet_processor_send     serial out to Teensy (U2D2 motors)
            └─> robot ─> packet_processor_receive ─> /motor_angle_feedback_topic ─> obs_sync

external inputs the pipeline waits on:
    /motor_angle_feedback_topic   17-dim encoder state; without it obs_sync publishes nothing
    /llm_state                    6-bit task state; idle pose until received
```

## Repository layout

```
repo: segmentation-guided-grasping
│
├── README.md
├── requirements.txt             # deploy: AI PC, numpy<2, CPU torch, pyserial
├── requirements_py312.txt       # py3.12 pinned variant
├── .gitignore                   # models/, *.pt, *.onnx, *.xml, *.bin, __pycache__/
│
└── src/act_yolo_grasp/          # the ROS 2 package (ament_python)
    ├── package.xml              # rclpy, std_msgs, sensor_msgs, cv_bridge,
    │                            # message_filters; exec_depend realsense2_camera,
    │                            # python3-serial
    ├── setup.py                 # installs launch/ + config/ to share/; 6 entry points:
    │                            # obs_sync, yolo_seg, act_policy, visualize_image,
    │                            # packet_processor_send, packet_processor_receive
    ├── setup.cfg
    ├── resource/
    │   └── act_yolo_grasp       # ament index marker (must match package name)
    │
    ├── config/
    │   ├── cameras.yaml         # 3x realsense serials + color profiles
    │   ├── obs_sync.yaml        # camera list, input/sync topics, slop, encoder timeout
    │   ├── yolo_seg.yaml        # IR dir path, device=intel:gpu, conf, input size,
    │   │                        # sync topics, action-done / water-grasp thresholds
    │   └── act_policy.yaml      # backend=openvino, ov_device=GPU, device=cpu,
    │                            # ckpt_dir/IR path, ACT hyperparams, topics, ensemble
    │
    ├── launch/
    │   ├── cameras.launch.py    # 3x realsense2_camera_node only (reusable alone)
    │   └── grasp_run.launch.py  # cameras + obs_sync + yolo_seg + act_policy + visualize
    │                            # (use_cameras:=false for bag replay, use_viz:=false headless;
    │                            # packet processors are run separately where the serial
    │                            # device is present)
    │
    ├── act_yolo_grasp/          # Python module (must match package name)
    │   ├── __init__.py
    │   ├── obs_sync_node.py     # RawObservationSyncNode: ApproximateTimeSynchronizer over
    │   │                        # 3 camera streams + /motor_angle_feedback_topic ->
    │   │                        # /sync/{cam}/image_raw + /sync/qpos
    │   ├── yolo_seg_node.py     # YoloSegNode: ultralytics YOLO on OpenVINO (no torch
    │   │                        # import); subscribes /llm_state (6-bit task state),
    │   │                        # locks/colors target instances, publishes
    │   │                        # /{cam}/YOLO_mask and /action_done when the target
    │   │                        # enters the basket (stateful LLM task orchestration)
    │   ├── task_logic.py        # pure ROS-free helpers extracted from yolo_seg_node:
    │   │                        # basket relabeling (mid/right), deterministic instance
    │   │                        # selection, in-basket overlap, convex-hull basket cover
    │   ├── act_policy_node.py   # ACT inference: synced obs + masks + /llm_state ->
    │   │                        # /motor_action_angle_topic; backend = openvino | onnx |
    │   │                        # torch; dataset stats normalization, temporal ensembling
    │   │                        # (+ ensemble-std diagnostic topic)
    │   ├── visualize_node.py    # MultiCameraViewer: OpenCV grid of camera feeds +
    │   │                        # YOLO masks with EMA-smoothed FPS overlays
    │   ├── packet_processor_send_node.py     # serial out to Teensy (/dev/teensy_serial):
    │   │                        # subscribes /motor_action_angle_topic, per-hand gripper
    │   │                        # thresholding + debounce, episode start/stop signals
    │   ├── packet_processor_receive_node.py  # serial in from Teensy: publishes
    │   │                        # /motor_angle_feedback_topic (17-dim encoder state),
    │   │                        # optional CSV angle logging
    │   └── act/                 # inference-only ACT subset (torch fallback):
    │       ├── __init__.py      #   policy.py + detr/models/* + detr/util/misc.py
    │       ├── policy.py                    # ACTPolicy (normalize + forward); also the ONNX export target
    │       └── detr/
    │           ├── __init__.py              # empty (exists in the original)
    │           ├── main.py                  # build_ACT_model_and_optimizer  [EDIT 1]
    │           ├── models/
    │           │   ├── __init__.py          # build_ACT_model / build_CNNMLP_model wrappers
    │           │   ├── detr_vae.py          # DETRVAE — the ACT encoder-decoder
    │           │   ├── backbone.py          # ResNet18 backbone           [EDIT 2]
    │           │   ├── transformer.py       # DETR transformer
    │           │   └── position_encoding.py # sine/learned pos-emb        [EDIT 2]
    │           └── util/
    │               ├── __init__.py
    │               └── misc.py              # only NestedTensor + is_main_process are used
    │
    ├── scripts/                 # tools, not ROS entry points
    │   ├── export_act_onnx.py   # ckpt -> ONNX (CPU-capable after the patch)
    │   ├── convert_to_openvino.py  # YOLO .pt -> IR dir; ACT .onnx -> .xml/.bin (FP16)
    │   └── test_llm_state.sh    # publish /llm_state test vectors by hand
    │
    ├── test/
    │   └── test_task_logic.py   # pytest for task_logic.py (synthetic masks)
    │
    └── models/                  # gitignored — weights live outside git
        ├── best_0615_for_mutiple_object.pt
        ├── best_yolo_mutiple_object_fp32_openvino_model/  # from convert_to_openvino.py yolo
        ├── policy_best.ckpt / .onnx
        ├── policy_best_openvino_model/                    # from convert_to_openvino.py act
        └── dataset_stats.pkl    # required next to the ACT model
```

## Bring-up order

1. `packet_processor_receive` + `packet_processor_send` on the machine with the
   Teensy serial device (`/dev/teensy_serial`) — encoder feedback must be
   flowing before obs_sync publishes anything.
2. `ros2 launch act_yolo_grasp grasp_run.launch.py` (add `use_cameras:=false`
   when replaying bags, `use_viz:=false` to run headless).
3. Publish `/llm_state` (6-bit task state) — e.g. via `scripts/test_llm_state.sh`;
   the policy holds an idle pose until the first state arrives.

## Model conversion

```
scripts/export_act_onnx.py       # ACT ckpt -> policy_best.onnx
scripts/convert_to_openvino.py   # YOLO .pt -> OpenVINO IR dir; ACT .onnx -> .xml/.bin (FP16)
```
