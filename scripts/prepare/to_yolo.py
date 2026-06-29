"""
Labelme JSON → YOLO 分割训练集（原图 + 裁剪双路）

用法:
    python -m scripts.prepare.to_yolo
    python -m scripts.prepare.to_yolo --split 0.8
    python -m scripts.prepare.to_yolo --holdout 30
"""

import argparse
import glob
import json
import os
import random
import shutil
from pathlib import Path

import cv2

from scripts.utils.paths import RAW_DIR, ANNOTATIONS_DIR, DATASET_DIR
from scripts.utils.crop import polygon_to_mask, polygon_bbox, crop_with_mask

IMG_EXTS = [".jpg", ".jpeg", ".png"]


def create_yolo_line(class_id, points, img_w, img_h):
    coords = []
    for x, y in points:
        # clamp to [0,1] — crop bounding boxes may slightly exceed polygon bounds
        coords.append(f"{max(0.0, min(1.0, x / img_w)):.6f}")
        coords.append(f"{max(0.0, min(1.0, y / img_h)):.6f}")
    return f"{class_id} " + " ".join(coords)


def crop_spine_and_make_label(img, points, spine_idx, basename, out_img_dir, out_label_dir, class_id=0):
    h, w = img.shape[:2]
    bbox = polygon_bbox(points)
    mask = polygon_to_mask(points, h, w)
    img_crop = crop_with_mask(img, mask, bbox, alpha=False)
    if img_crop is None:
        return

    x1, y1, _, _ = bbox
    crop_h, crop_w = img_crop.shape[:2]
    img_name = f"{basename}_crop_{spine_idx:03d}.jpg"
    cv2.imwrite(os.path.join(out_img_dir, img_name), img_crop)

    new_points = [(px - x1, py - y1) for px, py in points]
    yolo_line = create_yolo_line(class_id, new_points, crop_w, crop_h)
    label_path = os.path.join(out_label_dir, f"{basename}_crop_{spine_idx:03d}.txt")
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(yolo_line + "\n")


def build_dataset(image_dir=None, json_dir=None, output_dir=None, split_ratio=0.8, seed=42, no_crop=False, holdout=0):
    image_dir = Path(image_dir) if image_dir else RAW_DIR
    json_dir = Path(json_dir) if json_dir else ANNOTATIONS_DIR
    output_dir = Path(output_dir) if output_dir else DATASET_DIR

    sub_dirs = ["images/train", "images/val", "labels/train", "labels/val"]
    if holdout > 0:
        sub_dirs += ["images/test", "labels/test"]
    for sub in sub_dirs:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"), key=lambda f: int(f.stem))
    if not json_files:
        print("No JSON annotation files found")
        return

    random.seed(seed)
    random.shuffle(json_files)

    test_files = []
    if holdout > 0:
        test_files = json_files[:holdout]
        json_files = json_files[holdout:]

    split_idx = int(len(json_files) * split_ratio)
    splits = {"train": json_files[:split_idx], "val": json_files[split_idx:]}

    stats = {"original": 0, "crops": 0, "test_original": 0}
    splits["test"] = test_files
    for split_name, files in splits.items():
        if not files:
            continue
        img_dir_out = output_dir / "images" / split_name
        label_dir_out = output_dir / "labels" / split_name

        for json_path in files:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            basename = json_path.stem
            img_w = data.get("imageWidth", 0)
            img_h = data.get("imageHeight", 0)

            src_img = None
            for ext in IMG_EXTS:
                candidate = image_dir / f"{basename}{ext}"
                if candidate.exists():
                    src_img = candidate
                    break

            if src_img is None:
                img_data = data.get("imageData")
                if img_data:
                    import base64
                    img_bytes = base64.b64decode(img_data)
                    src_img = image_dir / f"{basename}.jpg"
                    src_img.write_bytes(img_bytes)
                else:
                    print(f"[skip] no image: {basename}")
                    continue

            shutil.copy2(str(src_img), str(img_dir_out / src_img.name))

            lines = [create_yolo_line(0, s["points"], img_w, img_h) for s in data["shapes"]]
            (label_dir_out / f"{basename}.txt").write_text("\n".join(lines), encoding="utf-8")
            stats["original"] += 1
            if split_name == "test":
                stats["test_original"] += 1

            if not no_crop:
                img = cv2.imread(str(src_img))
                if img is not None:
                    for i, shape in enumerate(data["shapes"]):
                        crop_spine_and_make_label(img, shape["points"], i, basename, str(img_dir_out), str(label_dir_out))
                        stats["crops"] += 1

    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(
        f"# YOLO train config\n"
        f"path: {output_dir.resolve()}\n"
        f"train: {(output_dir / 'images' / 'train').resolve()}\n"
        f"val: {(output_dir / 'images' / 'val').resolve()}\n\n"
        f"nc: 1\nnames:\n  0: book\n",
        encoding="utf-8",
    )

    if holdout > 0:
        test_yaml_path = output_dir / "test.yaml"
        test_yaml_path.write_text(
            f"# YOLO holdout test config (independent of train/val)\n"
            f"path: {output_dir.resolve()}\n"
            f"train: {(output_dir / 'images' / 'test').resolve()}\n"
            f"val: {(output_dir / 'images' / 'test').resolve()}\n\n"
            f"nc: 1\nnames:\n  0: book\n",
            encoding="utf-8",
        )

    print(f"\n===== Done =====")
    print(f"Original images: {stats['original']}")
    print(f"Cropped spines:  {stats['crops']}")
    print(f"Total images:    {stats['original'] + stats['crops']}")
    if holdout > 0:
        print(f"Holdout test:    {stats['test_original']} originals (excluded from train/val)")
    print(f"Train/Val split: {int(split_ratio * 100)}/{int((1 - split_ratio) * 100)}")
    print(f"Output:          {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Labelme JSON -> YOLO segmentation dataset")
    parser.add_argument("--image_dir", default=None, help="Original images dir (default: data/raw)")
    parser.add_argument("--json_dir", default=None, help="Labelme JSON dir (default: data/annotations)")
    parser.add_argument("--output_dir", default=None, help="Output dir (default: output/dataset)")
    parser.add_argument("--split", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--holdout", type=int, default=0, help="Number of original images to hold out as independent test set (excluded from train/val)")
    parser.add_argument("--no-crop", action="store_true", help="Skip crop augmentation, use full images only")
    args = parser.parse_args()
    build_dataset(args.image_dir, args.json_dir, args.output_dir, args.split, no_crop=args.no_crop, holdout=args.holdout)