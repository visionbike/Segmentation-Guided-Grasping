import json, glob, yaml, collections
from pathlib import Path

ROOT = Path("dataset_yolo").resolve()

cats = json.load(open("coco_json/train.json"))["categories"]
# convert_coco(cls91to80=False) writes cls = category_id - 1
names = {c["id"] - 1: c["name"] for c in cats if c["id"] - 1 >= 0}

cfg = {
    "path": str(ROOT),
    "train": "images/train",
    "val":   "images/val",
    "test":  "images/test",
    "names": {i: names[i] for i in sorted(names)},
}
(ROOT / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

used = {int(l.split()[0]) for f in glob.glob("dataset_yolo/labels/*/*.txt") for l in open(f) if l.strip()}
print("declared:", sorted(names), "\nused    :", sorted(used))
assert used <= set(names), f"labels reference undeclared classes: {used - set(names)}"
