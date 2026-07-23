"""Re-split dataset_yolo by capture group so no source scene straddles train/val/test.

The current split is frame-level random: all 16 video clips appear in all three
splits, so val/test are near-duplicates of train at 30fps. This groups every file
by its capture unit (video clip, timestamp burst, or index sequence) and assigns
whole groups.

Note on stems: 162 stems name 2-4 files each, and those files are *different
images* (verified: 0 of 1564 same-stem pairs match visually). The source upload
had colliding filenames across directories and roboflow kept them apart with the
.rf.<hash> suffix. Since provenance can't be recovered from a collided name, all
files sharing a stem are grouped together -- over-grouping costs a little split
granularity but cannot introduce leakage.
"""

import argparse
import collections
import datetime
import hashlib
import re
import shutil
from pathlib import Path

import yaml

SRC = Path("dataset_yolo")
SPLITS = ("train", "val", "test")
RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}
GAP = datetime.timedelta(seconds=30)   # burst boundary for timestamped families
RF = re.compile(r"\.rf\.[0-9a-f]+$")   # roboflow export hash


def stem_of(path):
    """Filename with the roboflow export hash stripped; colliding names share a stem."""
    return RF.sub("", path.stem)


def timestamp(stem):
    """(datetime, family) for the two timestamped families, else (None, None)."""
    m = re.match(r"photo_(\d{8})_(\d{6})_png$", stem)
    if m:
        return datetime.datetime.strptime(m[1] + m[2], "%Y%m%d%H%M%S"), "photo_ts"
    m = re.match(r"(\d{8})_(\d{6})_(\d{3})_png$", stem)
    if m:
        t = datetime.datetime.strptime(m[1] + m[2], "%Y%m%d%H%M%S")
        return t + datetime.timedelta(milliseconds=int(m[3])), "ts_ms"
    return None, None


def group_keys(stems):
    """Map each stem to a capture-group key.

    Video frames group by clip id. Timestamped stills cluster by GAP. The three
    index sequences are contiguous with no breaks, so each is a single group.
    Anything unrecognised becomes its own group, which is the conservative choice.
    """
    keys, pending = {}, collections.defaultdict(list)

    for stem in stems:
        m = re.match(r"(\d{8}_\d{6})_frame_\d+_jpg$", stem)
        if m:
            keys[stem] = f"clip:{m[1]}"
            continue
        t, family = timestamp(stem)
        if family:
            pending[family].append((t, stem))
            continue
        if re.match(r"rgb_\d{4}_png$", stem):
            keys[stem] = "seq:rgb"
        elif re.match(r"photo_\d{4}_png$", stem):
            keys[stem] = "seq:photo4"
        elif re.match(r"photo_\d{1,2}_jpg$", stem):
            keys[stem] = "seq:photoN"
        else:
            keys[stem] = f"solo:{stem}"

    for family, items in pending.items():
        items.sort()
        cluster = 0
        for i, (t, stem) in enumerate(items):
            if i and t - items[i - 1][0] > GAP:
                cluster += 1
            keys[stem] = f"{family}:{cluster:03d}"
    return keys


def collect():
    """Index every image, attach labels and classes.

    The md5 pass is a guard, not a cleanup step: the current export contains no
    byte-identical files. It exists so a future re-export with real duplicates
    does not silently over-weight those samples.
    """
    by_stem = collections.defaultdict(list)
    for split in SPLITS:
        for img in sorted((SRC / "images" / split).iterdir()):
            by_stem[stem_of(img)].append(img)

    files, dropped = [], 0
    for stem, paths in sorted(by_stem.items()):
        seen = {}
        for img in paths:
            digest = hashlib.md5(img.read_bytes()).hexdigest()
            if digest in seen:
                dropped += 1          # exact re-export, keep one copy
                continue
            seen[digest] = img
            label = SRC / "labels" / img.parent.name / f"{img.stem}.txt"
            classes = collections.Counter()
            if label.exists():
                for line in label.read_text().splitlines():
                    if line.strip():
                        classes[int(line.split()[0])] += 1
            files.append({"stem": stem, "img": img,
                          "label": label if label.exists() else None,
                          "classes": classes})
    return files, dropped


def assign(groups, names):
    """Greedy largest-first assignment minimising per-class and per-image imbalance."""
    total = collections.Counter()
    for g in groups.values():
        total.update(g["classes"])
    n_images = sum(g["n"] for g in groups.values())

    state = {s: {"classes": collections.Counter(), "n": 0} for s in SPLITS}

    def cost(split, g):
        """Total imbalance over ALL splits if g goes to `split`.

        Scoring only the receiving split is wrong: from an empty state a small
        group always looks cheaper in the 10% split than in the 70% one, so the
        large split never gets fed. The objective has to be global.
        """
        err = 0.0
        for s in SPLITS:
            gets = s == split
            for c in names:
                if not total[c]:
                    continue
                cur = state[s]["classes"][c] + (g["classes"][c] if gets else 0)
                err += (cur / total[c] - RATIOS[s]) ** 2
            cur_n = state[s]["n"] + (g["n"] if gets else 0)
            err += 2.0 * (cur_n / n_images - RATIOS[s]) ** 2
        return err

    # deterministic: largest group first, ties broken by key
    order = sorted(groups.items(), key=lambda kv: (-kv[1]["n"], kv[0]))
    plan = {}
    for key, g in order:
        best = min(SPLITS, key=lambda s: cost(s, g))
        plan[key] = best
        state[best]["classes"].update(g["classes"])
        state[best]["n"] += g["n"]

    # every class must appear in every split, or the split cannot score it
    for split in ("val", "test"):
        for c in names:
            if state[split]["classes"][c] or not total[c]:
                continue
            donors = [k for k, s in plan.items()
                      if s == "train" and groups[k]["classes"][c]
                      and all(state["train"]["classes"][cc] - groups[k]["classes"][cc] > 0
                              for cc in groups[k]["classes"])]
            if not donors:
                print(f"  ! class {c} ({names[c]}) cannot be placed in {split}")
                continue
            donor = min(donors, key=lambda k: groups[k]["n"])
            plan[donor] = split
            state[split]["classes"].update(groups[donor]["classes"])
            state[split]["n"] += groups[donor]["n"]
            state["train"]["classes"].subtract(groups[donor]["classes"])
            state["train"]["n"] -= groups[donor]["n"]
            print(f"  moved {donor} -> {split} to cover class {c} ({names[c]})")
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("dataset_yolo_split"))
    ap.add_argument("--link", action="store_true", help="symlink instead of copy")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    names = yaml.safe_load((SRC / "data.yaml").read_text())["names"]

    files, dropped = collect()
    print(f"files kept {len(files)}, exact duplicates dropped {dropped}")

    keys = group_keys({f["stem"] for f in files})
    groups = collections.defaultdict(lambda: {"n": 0, "classes": collections.Counter(),
                                              "files": []})
    for f in files:
        g = groups[keys[f["stem"]]]
        g["n"] += 1
        g["classes"].update(f["classes"])
        g["files"].append(f)
    print(f"capture groups {len(groups)}")

    plan = assign(dict(groups), names)

    print(f"\n{'split':<8}{'groups':>8}{'images':>8}{'inst':>8}  per-class")
    for split in SPLITS:
        sel = [g for k, g in groups.items() if plan[k] == split]
        cls = collections.Counter()
        for g in sel:
            cls.update(g["classes"])
        n = sum(g["n"] for g in sel)
        print(f"{split:<8}{len(sel):>8}{n:>8}{sum(cls.values()):>8}  "
              + " ".join(f"{c}:{cls[c]}" for c in sorted(names)))

    if args.dry_run:
        return

    for split in SPLITS:
        for kind in ("images", "labels"):
            (args.out / kind / split).mkdir(parents=True, exist_ok=True)

    place = (lambda s, d: d.symlink_to(s.resolve())) if args.link else shutil.copy2
    for key, g in groups.items():
        split = plan[key]
        for f in g["files"]:
            place(f["img"], args.out / "images" / split / f["img"].name)
            if f["label"]:
                place(f["label"], args.out / "labels" / split / f["label"].name)

    cfg = {"path": str(args.out.resolve()), "train": "images/train",
           "val": "images/val", "test": "images/test",
           "names": {int(i): names[i] for i in sorted(names)}}
    (args.out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # audit: no stem and no capture group may cross a split boundary
    where = collections.defaultdict(set)
    for split in SPLITS:
        for img in (args.out / "images" / split).iterdir():
            where[stem_of(img)].add(split)
    straddling = {s for s, v in where.items() if len(v) > 1}
    gsplits = collections.defaultdict(set)
    for key, split in plan.items():
        gsplits[key].add(split)
    print(f"\nwrote {args.out}/data.yaml")
    print(f"stems in >1 split: {len(straddling)} (want 0)")
    print(f"groups in >1 split: {sum(1 for v in gsplits.values() if len(v) > 1)} (want 0)")


if __name__ == "__main__":
    main()
