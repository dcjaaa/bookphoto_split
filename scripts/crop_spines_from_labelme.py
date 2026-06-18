"""
标注完成后 → 自动裁剪每本书脊到 data/split/{n}/

用法:
    python scripts/crop_spines_from_labelme.py
    python scripts/crop_spines_from_labelme.py --image 5
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.paths import RAW_DIR, ANNOTATIONS_DIR, SPLIT_DIR


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
    out_dir = SPLIT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cropped = 0
    for shape in shapes:
        pts = np.array(shape["points"], dtype=np.int32)
        x1, y1 = max(0, int(pts[:, 0].min())), max(0, int(pts[:, 1].min()))
        x2, y2 = min(w, int(pts[:, 0].max())), min(h, int(pts[:, 1].max()))
        if x2 <= x1 or y2 <= y1:
            continue

        full_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(full_mask, [pts], 255)
        mask_crop = full_mask[y1:y2, x1:x2]
        img_crop = img[y1:y2, x1:x2]
        img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)

        b, g, r = cv2.split(img_crop)
        alpha = mask_crop
        result = cv2.merge([b, g, r, alpha])

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
    subdirs = [d for d in SPLIT_DIR.iterdir() if d.is_dir()]
    print(f"\nDone: {len(json_files)} images -> {total} spines | {len(subdirs)} subdirs in {SPLIT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-crop book spines from Labelme annotations")
    parser.add_argument("--image", type=int, default=None, help="Only crop image N")
    args = parser.parse_args()
    crop_all(args.image)