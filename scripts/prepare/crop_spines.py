"""
标注完成后 → 自动裁剪每本书脊到 data/crops_labelme/{n}/

用法:
    python -m scripts.prepare.crop_spines
    python -m scripts.prepare.crop_spines --image 5
"""

import argparse
import json
from pathlib import Path

import cv2

from scripts.utils.paths import RAW_DIR, ANNOTATIONS_DIR, CROPS_LABELED_DIR
from scripts.utils.crop import polygon_to_mask, polygon_bbox, crop_with_mask


def crop_one_image(json_path: Path) -> int:
    name = json_path.stem
    img_path = None
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = RAW_DIR / f"{name}{ext}"
        if candidate.exists():
            img_path = candidate
            break
    if img_path is None:
        print(f"  [skip] no image for {json_path.name}")
        return 0

    data = json.loads(json_path.read_text(encoding="utf-8"))
    shapes = data.get("shapes", [])
    if not shapes:
        print(f"  [empty] {json_path.name}")
        return 0

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [skip] cannot read {img_path}")
        return 0

    h, w = img.shape[:2]
    out_dir = CROPS_LABELED_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cropped = 0
    for shape in shapes:
        points = shape["points"]
        bbox = polygon_bbox(points)
        mask = polygon_to_mask(points, h, w)
        result = crop_with_mask(img, mask, bbox, alpha=True)
        if result is None:
            continue
        out_path = out_dir / f"spine_{cropped:03d}.png"
        cv2.imwrite(str(out_path), result)
        cropped += 1

    print(f"  {name}: {cropped} spines -> {out_dir}/")
    return cropped


def crop_all(only_image: int | None = None):
    if only_image is not None:
        json_path = ANNOTATIONS_DIR / f"{only_image}.json"
        if json_path.exists():
            crop_one_image(json_path)
        else:
            print(f"No annotation found: {json_path}")
        return

    json_files = sorted(ANNOTATIONS_DIR.glob("*.json"), key=lambda f: int(f.stem))
    total = sum(crop_one_image(f) for f in json_files)
    subdirs = [d for d in CROPS_LABELED_DIR.iterdir() if d.is_dir()]
    print(f"\nDone: {len(json_files)} images -> {total} spines | {len(subdirs)} subdirs in {CROPS_LABELED_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-crop book spines from Labelme annotations")
    parser.add_argument("--image", type=int, default=None, help="Only crop image N")
    args = parser.parse_args()
    crop_all(args.image)
