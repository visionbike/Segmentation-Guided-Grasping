  repo: segmentation-guided-grasping        (branch: intel-deployment)
  │
  ├── README.md                    # which machine uses which requirements file,
  │                                # bring-up order, model conversion steps
  ├── requirements.txt             # deploy: AI PC, py3.12, numpy<2, CPU torch, pyserial
  ├── requirements-dev.txt         # dev: conda env, py3.14, numpy>=2.3
  ├── .gitignore                   # models/, *.pt, *.onnx, *.xml, *.bin, __pycache__/
  │
  └── act_yolo_grasp/              # the ROS 2 package (ament_python)
      ├── package.xml              # <name>act_yolo_grasp</name>; rclpy, std_msgs,
      │                            # sensor_msgs, cv_bridge, message_filters;
      │                            # exec_depend realsense2_camera
      ├── setup.py                 # package_name = 'act_yolo_grasp'; installs
      │                            # launch/ + config/ to share/; 5 entry points
      ├── setup.cfg
      ├── resource/
      │   └── act_yolo_grasp       # ament index marker (must match package name)
      │
      ├── config/
      │   ├── cameras.yaml         # 3x realsense serials + color profiles
      │   ├── yolo_seg.yaml        # IR dir path, device=intel:gpu, conf, input size,
      │   │                        # sync topics, action-done / water-grasp thresholds
      │   └── act_policy.yaml      # backend=openvino, ov_device=GPU, device=cpu,
      │                            # ckpt_dir/IR path, ACT hyperparams, topics, ensemble
      │
      ├── launch/
      │   ├── cameras.launch.py    # 3x realsense2_camera_node only (reusable alone)
      │   └── grasp_run.launch.py  # cameras + obs_sync + yolo_seg + act_policy + visualize
      │
      ├── act_yolo_grasp/          # Python module (must match package name)
      │   ├── __init__.py
      │   ├── obs_sync_node.py     # from obs_sync_show.py
      │   ├── yolo_seg_node.py     # from YOLO_segmentation_LLM_scenerio_show.py,
      │   │                        # OpenVINO via ultralytics, no torch import
      │   ├── act_policy_node.py   # from run_ACT_policy_LLM_scenerio.py,
      │   │                        # backend = openvino | onnx | torch
      │   ├── visualize_node.py    # from visualize_image.py
      │   ├── task_logic.py        # optional: basket relabel / action-done helpers
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
      │   └── convert_to_openvino.py  # YOLO .pt -> IR dir; ACT .onnx -> .xml/.bin (FP16)
      │
      └── models/                  # gitignored — weights live outside git
          ├── best_0613.pt
          ├── best_0613_openvino_model/       # from convert_to_openvino.py yolo
          ├── policy_best.ckpt / .onnx
          ├── policy_best.xml / .bin          # from convert_to_openvino.py act
          └── dataset_stats.pkl               # required next to the ACT model
