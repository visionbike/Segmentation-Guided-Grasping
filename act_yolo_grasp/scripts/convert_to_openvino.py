"""Convert models to OpenVINO IR (FP16) for the Intel AI PC (Arc GPU / NPU).

  YOLO:  ultralytics .pt  -> <name>_openvino_model/  (IR FP16, dynamic batch)
  ACT :  .onnx -> <name>.xml/.bin  (IR FP16)
         a .ckpt is also accepted: it is exported to ONNX first via
         scripts/export_act_onnx.py (CPU is sufficient).

Examples (from the package root):
    python3 scripts/convert_to_openvino.py yolo --pt models/best_0613.pt --imgsz 320
    python3 scripts/convert_to_openvino.py act  --onnx models/policy_best.onnx --verify
    python3 scripts/convert_to_openvino.py act  --ckpt_dir models --verify --verify_device GPU
"""

import argparse
import os
import subprocess
import sys


def convert_yolo(pt_path, imgsz, half=True):
    from ultralytics import YOLO

    model = YOLO(pt_path)
    # dynamic=True keeps the batch axis dynamic so yolo_seg_node can keep
    # batching 3 camera frames per call on the Arc GPU.
    # For an NPU-only deployment re-export with dynamic=False, batch=1.
    out_dir = model.export(
        format="openvino",
        half=half,          # FP16 weights
        imgsz=imgsz,        # 320 matches the node's model(batch_imgs, imgsz=320)
        dynamic=True,
        nms=False,
    )
    print(f"[OK] YOLO OpenVINO IR: {out_dir}")
    return str(out_dir)


def convert_act_onnx_to_ir(onnx_path, output=None):
    import openvino as ov

    output = output or os.path.splitext(onnx_path)[0] + ".xml"
    model = ov.convert_model(onnx_path)
    # compress_to_fp16=True stores weights as f16; execution precision is
    # chosen per device at compile time.
    ov.save_model(model, output, compress_to_fp16=True)
    print(f"[OK] ACT OpenVINO IR: {output} (+ .bin)")
    return output


def export_act_ckpt_to_onnx(ckpt_dir, ckpt_name, extra_args):
    """Run scripts/export_act_onnx.py (CPU-capable)."""
    onnx_path = os.path.join(ckpt_dir, os.path.splitext(ckpt_name)[0] + ".onnx")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_act_onnx.py")
    subprocess.run(
        [sys.executable, script, "--ckpt_dir", ckpt_dir, "--ckpt_name", ckpt_name,
         "--output", onnx_path] + extra_args,
        check=True,
    )
    return onnx_path


def verify_act(onnx_path, xml_path, ncam, height, width, qpos_dim, device="CPU"):
    """Compare onnxruntime (CPU, FP32) against the FP16 IR on the chosen device."""
    import numpy as np
    import onnxruntime as ort
    import openvino as ov

    rng = np.random.default_rng()
    qpos = rng.standard_normal((1, qpos_dim), dtype=np.float32)
    image = rng.random((1, ncam, 3, height, width), dtype=np.float32)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ref = np.asarray(session.run(["action"], {"qpos": qpos, "image": image})[0])

    core = ov.Core()
    compiled = core.compile_model(core.read_model(xml_path), device)
    out = np.asarray(compiled({"qpos": qpos, "image": image})["action"])

    diff = float(np.abs(ref - out).max())
    # FP16 tolerance: actions are normalized (~unit scale), 1e-2 is safe.
    status = "PASS" if diff < 1e-2 else "CHECK (fp16 rounding vs export error?)"
    print(f"[VERIFY] device={device} max_abs_diff={diff:.4e} -> {status}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_yolo = sub.add_parser("yolo", help="ultralytics .pt -> OpenVINO IR dir")
    p_yolo.add_argument("--pt", required=True, help="path to the YOLO .pt weights")
    p_yolo.add_argument("--imgsz", type=int, default=320)

    p_act = sub.add_parser("act", help="ACT .onnx (or .ckpt) -> OpenVINO IR")
    p_act.add_argument("--onnx", help="existing ONNX (recommended input)")
    p_act.add_argument("--ckpt_dir", help="fallback: export ONNX from this checkpoint dir first")
    p_act.add_argument("--ckpt_name", default="policy_best.ckpt")
    p_act.add_argument("--output", help="output .xml path (default: alongside the ONNX)")
    p_act.add_argument("--verify", action="store_true")
    p_act.add_argument("--verify_device", default="CPU", help="CPU | GPU | NPU")
    p_act.add_argument("--ncam", type=int, default=3)
    p_act.add_argument("--height", type=int, default=240)
    p_act.add_argument("--width", type=int, default=320)
    p_act.add_argument("--qpos_dim", type=int, default=17)

    args, extra = parser.parse_known_args()

    if args.cmd == "yolo":
        convert_yolo(args.pt, args.imgsz)
        return

    onnx_path = args.onnx
    if onnx_path is None:
        if not args.ckpt_dir:
            parser.error("act: provide --onnx or --ckpt_dir")
        onnx_path = export_act_ckpt_to_onnx(args.ckpt_dir, args.ckpt_name, extra)

    xml_path = convert_act_onnx_to_ir(onnx_path, args.output)

    if args.verify:
        verify_act(onnx_path, xml_path, args.ncam, args.height, args.width,
                   args.qpos_dim, args.verify_device)


if __name__ == "__main__":
    main()
