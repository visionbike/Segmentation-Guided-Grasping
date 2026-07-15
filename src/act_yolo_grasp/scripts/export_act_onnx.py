
"""Export an ACTPolicy checkpoint (act_yolo_grasp.act) to ONNX.

Reproduces exactly the inference contract used by act_policy_node.py:
  - qpos: (B, obs_dim) float32, ALREADY normalized with dataset_stats.pkl's
          qpos_mean / qpos_std (raw robot joint state, no extra "task" dims).
  - image: (B, num_cam, 3, H, W) float32 in [0, 1], RGB, resized to (W, H).
           ImageNet normalization is applied INSIDE the exported graph
           (ACTPolicy.__call__ does this), so do not normalize the image yourself.
  - output: (B, chunk_size, action_dim) raw (still normalized) action chunk;
            denormalize afterwards with action_mean / action_std, same as
            post_process_action() in act_policy_node.py.

Important limitation: the number/order of cameras and the image size are baked
into the graph at export time (the model loops over camera_names in Python);
only the batch dimension is exported as dynamic.

Runs on CPU or CUDA (the vendored detr/main.py only moves the model to CUDA
when it is available). CPU export produces an identical graph, just slower.

Example (from the package root):
    python3 scripts/export_act_onnx.py \\
        --ckpt_dir models --ckpt_name policy_best.ckpt \\
        --camera_names top left_wrist right_wrist --chunk_size 10

    # fixed batch=1 graph for GPU/NPU (matches the node's runtime reshape):
    python3 scripts/export_act_onnx.py \\
        --ckpt_dir models --ckpt_name policy_best.ckpt \\
        --camera_names top left_wrist right_wrist --static-batch
"""

import argparse
import pickle
import sys
from pathlib import Path
import numpy as np
import torch

# make the package importable when running from the source tree
# (scripts/ -> package root, which contains the act_yolo_grasp module dir)
_PKG_ROOT = str(Path(__file__).resolve().parent.parent)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from act_yolo_grasp.act import ACTPolicy  # noqa: E402


class ACTExportWrapper(torch.nn.Module):
    """Thin wrapper exposing a plain forward() for tracing/export."""

    def __init__(self, act_policy):
        super().__init__()
        self.act_policy = act_policy

    def forward(self, qpos, image):
        return self.act_policy(qpos, image)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="Directory containing the ACT checkpoint and dataset_stats.pkl")
    p.add_argument("--ckpt_name", type=str, default="policy_best.ckpt")
    p.add_argument("--stats_name", type=str, default="dataset_stats.pkl")
    p.add_argument("--output", type=str, default=None,
                   help="Output .onnx path (default: <ckpt_dir>/<ckpt_name stem>.onnx)")

    p.add_argument("--image_width", type=int, default=320)
    p.add_argument("--image_height", type=int, default=240)
    p.add_argument("--camera_names", type=str, nargs="+",
                   default=["top", "left_wrist", "right_wrist"])

    p.add_argument("--chunk_size", type=int, default=10, help="num_queries")
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dim_feedforward", type=int, default=3200)
    p.add_argument("--enc_layers", type=int, default=4)
    p.add_argument("--dec_layers", type=int, default=7)
    p.add_argument("--nheads", type=int, default=8)
    p.add_argument("--backbone", type=str, default="resnet18")
    p.add_argument("--sep_CNN", dest="sep_CNN", action="store_true", default=True)
    p.add_argument("--no_sep_CNN", dest="sep_CNN", action="store_false")
    p.add_argument("--obs_dim", type=int, default=17)
    p.add_argument("--action_dim", type=int, default=17)

    # unused at inference time, but ACTPolicy/build_ACT_model_and_optimizer require them
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--kl_weight", type=float, default=50)

    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Batch size for the dummy export trace (forced to 1 with --static-batch)")
    p.add_argument("--static-batch", dest="static_batch", action="store_true",
                   help="Export a fixed batch=1 graph (no dynamic axes). The runtime "
                        "is always batch=1, so this makes the IR inherently GPU/NPU-static.")
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-3)
    return p.parse_args()


def build_policy(args):
    policy_config = {
        "lr": args.lr,
        "num_queries": args.chunk_size,
        "kl_weight": args.kl_weight,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "lr_backbone": args.lr_backbone,
        "backbone": args.backbone,
        "enc_layers": args.enc_layers,
        "dec_layers": args.dec_layers,
        "nheads": args.nheads,
        "camera_names": args.camera_names,
        "sep_CNN": args.sep_CNN,
        "obs_dim": args.obs_dim,
        "action_dim": args.action_dim,
    }
    print("[INFO] Building ACTPolicy with config:")
    for k, v in policy_config.items():
        print(f"         {k}: {v}")

    policy = ACTPolicy(policy_config)

    ckpt_path = Path(args.ckpt_dir) / args.ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    load_status = policy.load_state_dict(state_dict, strict=True)
    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    print(f"[INFO] load_state_dict status: {load_status}")

    policy.eval()
    return policy


def sanity_check_stats(args):
    stats_path = Path(args.ckpt_dir) / args.stats_name
    if not stats_path.exists():
        print(f"[WARN] dataset_stats.pkl not found at {stats_path}, "
              f"skipping obs_dim/action_dim cross-check")
        return
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    qpos_dim = np.asarray(stats["qpos_mean"]).reshape(-1).shape[0]
    action_dim = np.asarray(stats["action_mean"]).reshape(-1).shape[0]
    if qpos_dim != args.obs_dim:
        print(f"[WARN] dataset_stats.pkl qpos_mean has dim {qpos_dim}, "
              f"but --obs_dim={args.obs_dim}")
    if action_dim != args.action_dim:
        print(f"[WARN] dataset_stats.pkl action_mean has dim {action_dim}, "
              f"but --action_dim={args.action_dim}")


def export(args):
    sanity_check_stats(args)
    if args.static_batch:
        args.batch_size = 1  # a fixed-shape graph is only valid at batch=1
    policy = build_policy(args)
    device = next(policy.parameters()).device
    # label from declared attrs (.type / .index): torch's stubs declare no
    # __str__ on torch.device, which trips IDE inspections
    device_label = device.type if device.index is None else f"{device.type}:{device.index}"
    print(f"[INFO] Model device: {device_label}")

    num_cam = len(args.camera_names)
    dummy_qpos = torch.randn(args.batch_size, args.obs_dim, device=device)
    dummy_image = torch.rand(args.batch_size, num_cam, 3,
                             args.image_height, args.image_width, device=device)

    wrapper = ACTExportWrapper(policy).to(device).eval()

    with torch.no_grad():
        torch_out = wrapper(dummy_qpos, dummy_image)
    expected_shape = (args.batch_size, args.chunk_size, args.action_dim)
    print(f"[INFO] Pre-export forward pass OK, output shape={tuple(torch_out.shape)} "
          f"(expected {expected_shape})")
    assert tuple(torch_out.shape) == expected_shape, \
        "Unexpected output shape, check policy_config values"

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.ckpt_dir) / (Path(args.ckpt_name).stem + ".onnx")
    output_path.resolve().parent.mkdir(parents=True, exist_ok=True)

    # Static batch=1 (no dynamic axes) vs dynamic batch axis. The ACT node
    # always infers batch=1 and reshapes the IR to [1, ...] at load, so static
    # matches runtime exactly and yields an inherently GPU/NPU-static IR.
    dynamic_axes = None if args.static_batch else {
        "qpos": {0: "batch"},
        "image": {0: "batch"},
        "action": {0: "batch"},
    }
    mode = "static batch=1" if args.static_batch else "dynamic batch"
    print(f"[INFO] Exporting to {output_path} (opset={args.opset}, {mode}) ...")
    torch.onnx.export(
        wrapper,
        (dummy_qpos, dummy_image),
        output_path,
        input_names=["qpos", "image"],
        output_names=["action"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        # ACTPolicy overrides __call__ (not forward), which the torch>=2.9
        # dynamo exporter cannot trace; the legacy TorchScript exporter can.
        dynamo=False,
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[INFO] Export done. File size: {size_mb:.2f} MB")

    verify(args, wrapper, output_path, device)
    return output_path


def verify(args, wrapper, output_path, device):
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("[WARN] onnx / onnxruntime not installed, skipping verification. "
              "Install with: pip install onnx onnxruntime")
        return

    print("[INFO] Running onnx.checker.check_model ...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("[INFO] onnx.checker passed.")

    num_cam = len(args.camera_names)
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

    # ONNXRuntime only has CPUExecutionProvider here, so compare against a CPU
    # torch forward pass too. Comparing cuda-torch vs cpu-onnxruntime conflates
    # export error with ordinary cuda/cpu floating-point divergence.
    wrapper_cpu = wrapper.to("cpu").eval()

    max_abs_diff = 0.0
    max_rel_diff = 0.0
    rel_diff_floor = 1e-2  # ignore near-zero outputs when computing relative error
    # a static graph only accepts batch=1; a dynamic one also gets batch 2/3 tested
    n_trials = 1 if args.static_batch else 3
    for trial in range(n_trials):
        batch = 1 if args.static_batch else max(1, args.batch_size + trial)
        qpos = torch.randn(batch, args.obs_dim)
        image = torch.rand(batch, num_cam, 3, args.image_height, args.image_width)

        with torch.no_grad():
            torch_out = wrapper_cpu(qpos, image).numpy()

        onnx_out = np.asarray(session.run(
            ["action"],
            {"qpos": qpos.numpy().astype(np.float32),
             "image": image.numpy().astype(np.float32)},
        )[0])

        abs_diff = np.abs(torch_out - onnx_out)
        rel_diff = abs_diff / np.maximum(np.abs(torch_out), rel_diff_floor)
        trial_abs = float(abs_diff.max())
        trial_rel = float(rel_diff.max())
        max_abs_diff = max(max_abs_diff, trial_abs)
        max_rel_diff = max(max_rel_diff, trial_rel)
        print(f"[INFO] trial {trial}: batch={batch}, "
              f"max_abs_diff={trial_abs:.3e}, max_rel_diff={trial_rel:.3e}")

    wrapper.to(device)  # restore for any caller that reuses `wrapper` after verify()

    ok = max_abs_diff <= args.atol or max_rel_diff <= args.rtol
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] PyTorch vs ONNXRuntime max_abs_diff={max_abs_diff:.3e}, "
          f"max_rel_diff={max_rel_diff:.3e} (atol={args.atol}, rtol={args.rtol})")
    if not ok:
        print("[WARN] Numerical mismatch exceeds tolerance. Try a different --opset, "
              "or inspect the ONNX graph for ops that fall back to different kernels "
              "(e.g. nn.MultiheadAttention).")


if __name__ == "__main__":
    export(parse_args())
