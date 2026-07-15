"""Convert models to OpenVINO IR (FP16) for the Intel AI PC (Arc GPU / NPU).

  YOLO:  ultralytics .pt   -> <name>_openvino_model/  (IR FP16, dynamic batch)
         ultralytics .onnx -> <name>_openvino_model/  (IR FP16 + metadata.yaml
         rebuilt from the ONNX so ultralytics recovers class names / task)
  ACT :  .onnx -> <name>_openvino_model/  (IR FP16 + metadata.yaml)
         a .ckpt is also accepted: it is exported to ONNX first via
         scripts/export_act_onnx.py (CPU is sufficient).

Examples (from the package root):
    python3 scripts/convert_to_openvino.py yolo --pt   models/best_0613.pt --imgsz 320
    python3 scripts/convert_to_openvino.py yolo --onnx models/best_0613_fp32.onnx
    python3 scripts/convert_to_openvino.py act  --onnx models/policy_best.onnx --verify
    python3 scripts/convert_to_openvino.py act  --ckpt_dir models --verify --verify_device GPU
"""

import argparse
import ast
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


def convert_yolo_from_onnx(onnx_path, half=True):
    """ultralytics YOLO .onnx -> <name>_openvino_model/ (IR FP16 + metadata.yaml).

    ov.convert_model alone produces a bare .xml/.bin with no metadata, and
    ultralytics' YOLO() loader then falls back to numeric class names, which
    breaks yolo_seg_node's `cls_name == "basket"` logic. The class names /
    task / stride / imgsz are embedded in the ONNX metadata_props (ultralytics
    exports them there), so we rebuild metadata.yaml from those.
    """
    import onnx
    import openvino as ov
    import yaml

    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(onnx_path)),
                           f"{stem}_openvino_model")
    os.makedirs(out_dir, exist_ok=True)
    xml_path = os.path.join(out_dir, f"{stem}.xml")

    model = ov.convert_model(onnx_path)
    ov.save_model(model, xml_path, compress_to_fp16=half)

    props = {p.key: p.value for p in onnx.load(onnx_path, load_external_data=False).metadata_props}
    if "names" not in props:
        raise ValueError(
            f"{onnx_path} has no embedded 'names' metadata - it was not exported "
            f"by ultralytics. Re-export from the .pt with format='onnx', or use --pt."
        )
    metadata = {
        "description": props.get("description", ""),
        "author": props.get("author", "Ultralytics"),
        "date": props.get("date", ""),
        "version": props.get("version", ""),
        "license": props.get("license", ""),
        "docs": props.get("docs", ""),
        "stride": int(props.get("stride", 32)),
        "task": props.get("task", "segment"),
        "batch": int(props.get("batch", 1)),
        "imgsz": ast.literal_eval(props["imgsz"]),   # e.g. "[320, 320]"
        "names": ast.literal_eval(props["names"]),   # e.g. "{0: 'basket', ...}"
    }
    with open(os.path.join(out_dir, "metadata.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False, allow_unicode=True)

    print(f"[OK] YOLO OpenVINO IR: {out_dir}  ({len(metadata['names'])} classes)")
    return out_dir


def convert_act_onnx_to_ir(onnx_path, output_dir=None):
    """ACT .onnx -> <name>_openvino_model/ (IR FP16 + metadata.yaml).

    Mirrors the YOLO layout: a self-contained folder with <name>.xml, <name>.bin
    and metadata.yaml. metadata.yaml documents the shape contract read from the
    ONNX (camera_names are not encoded in the graph, only num_cam). Returns the
    .xml path inside the folder.
    """
    import onnx
    import openvino as ov
    import yaml

    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    out_dir = output_dir or os.path.join(
        os.path.dirname(os.path.abspath(onnx_path)), f"{stem}_openvino_model")
    os.makedirs(out_dir, exist_ok=True)
    xml_path = os.path.join(out_dir, f"{stem}.xml")

    model = ov.convert_model(onnx_path)
    # compress_to_fp16=True stores weights as f16; execution precision is
    # chosen per device at compile time.
    ov.save_model(model, xml_path, compress_to_fp16=True)

    def _shape(io):
        return [d.dim_value if d.dim_value else d.dim_param
                for d in io.type.tensor_type.shape.dim]

    g = onnx.load(onnx_path, load_external_data=False).graph
    inp = {i.name: _shape(i) for i in g.input}
    out = {o.name: _shape(o) for o in g.output}
    qpos, image, action = inp.get("qpos", []), inp.get("image", []), out.get("action", [])
    metadata = {
        "task": "act_policy",
        "precision": "FP16",
        "static_batch": bool(qpos) and all(isinstance(d, int) for d in qpos),
        "obs_dim": qpos[-1] if qpos else None,
        "num_cam": image[1] if len(image) >= 2 else None,
        "image_height": image[-2] if len(image) >= 2 else None,
        "image_width": image[-1] if len(image) >= 1 else None,
        "chunk_size": action[1] if len(action) >= 2 else None,
        "action_dim": action[-1] if action else None,
        "note": "camera_names come from the ROS param; only num_cam is encoded in the graph",
    }
    with open(os.path.join(out_dir, "metadata.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)

    print(f"[OK] ACT OpenVINO IR: {out_dir}")
    return xml_path


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

    p_yolo = sub.add_parser("yolo", help="ultralytics .pt or .onnx -> OpenVINO IR dir")
    g_yolo = p_yolo.add_mutually_exclusive_group(required=True)
    g_yolo.add_argument("--pt", help="path to the YOLO .pt weights")
    g_yolo.add_argument("--onnx", help="path to an ultralytics-exported YOLO .onnx")
    p_yolo.add_argument("--imgsz", type=int, default=320, help="only used with --pt")

    p_act = sub.add_parser("act", help="ACT .onnx (or .ckpt) -> OpenVINO IR")
    p_act.add_argument("--onnx", help="existing ONNX (recommended input)")
    p_act.add_argument("--ckpt_dir", help="fallback: export ONNX from this checkpoint dir first")
    p_act.add_argument("--ckpt_name", default="policy_best.ckpt")
    p_act.add_argument("--output", help="output IR folder (default: <onnx_stem>_openvino_model/ beside the ONNX)")
    p_act.add_argument("--verify", action="store_true")
    p_act.add_argument("--verify_device", default="CPU", help="CPU | GPU | NPU")
    p_act.add_argument("--ncam", type=int, default=3)
    p_act.add_argument("--height", type=int, default=240)
    p_act.add_argument("--width", type=int, default=320)
    p_act.add_argument("--qpos_dim", type=int, default=17)

    args, extra = parser.parse_known_args()

    if args.cmd == "yolo":
        if args.pt:
            convert_yolo(args.pt, args.imgsz)
        else:
            convert_yolo_from_onnx(args.onnx)
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
