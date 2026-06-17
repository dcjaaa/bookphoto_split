"""
标注完成后 → 自动裁剪每本书脊到 data/split/{n}/

目录结构:
    data/
    ├── raw/                 ← 原始照片
    ├── annotations/         ← Labelme标注JSON
    │   ├── 1.json
    │   └── 2.json
    └── split/               ← 裁剪后的单独书脊图片
        ├── 1/
        │   ├── spine_000.jpg
        │   └── spine_001.jpg
        └── 2/
            └── ...

用法:
    python scripts/auto_crop_annotations.py              # 裁剪全部
    python scripts/auto_crop_annotations.py --image 5    # 只裁剪第5张
"""
import os
import json
import cv2
import numpy as np
import argparse

RAW_DIR = "data/raw"
ANNOTATION_DIR = "data/annotations"
SPLIT_DIR = "data/split"


def crop_one_image(json_filename):
    """裁剪单张图片的所有标注书脊"""
    json_path = os.path.join(ANNOTATION_DIR, json_filename)
    name = os.path.splitext(json_filename)[0]

    # 找对应原图
    img_path = None
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = os.path.join(RAW_DIR, name + ext)
        if os.path.exists(candidate):
            img_path = candidate
            break
    if img_path is None:
        print(f"[skip] no image for {json_filename}")
        return 0

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    if not shapes:
        print(f"[empty] {json_filename}")
        return 0

    img = cv2.imread(img_path)
    if img is None:
        print(f"[skip] cannot read {img_path}")
        return 0

    h, w = img.shape[:2]

    # 输出到 data/split/{name}/
    out_dir = os.path.join(SPLIT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    cropped = 0
    for i, shape in enumerate(shapes):
        pts = np.array(shape["points"], dtype=np.int32)

        x1 = max(0, int(pts[:, 0].min()))
        y1 = max(0, int(pts[:, 1].min()))
        x2 = min(w, int(pts[:, 0].max()))
        y2 = min(h, int(pts[:, 1].max()))

        if x2 <= x1 or y2 <= y1:
            continue

        # 全图 mask 然后裁剪
        full_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(full_mask, [pts], 255)

        # 裁剪 mask 和图片到同一 bbox
        mask_crop = full_mask[y1:y2, x1:x2]
        img_crop = img[y1:y2, x1:x2]

        # 多边形外变黑，保存为 jpg (jpg 无透明通道，用黑色代替)
        img_crop = cv2.bitwise_and(img_crop, img_crop, mask=mask_crop)
        # 把透明的黑色变成 PNG 保留透明
        b, g, r = cv2.split(img_crop)
        alpha = mask_crop
        result = cv2.merge([b, g, r, alpha])

        out_path = os.path.join(out_dir, f"spine_{cropped:03d}.png")
        cv2.imwrite(out_path, result)
        cropped += 1

    print(f"  {name}: {cropped} spines -> {out_dir}/")
    return cropped


def crop_all(only_image=None):
    if only_image:
        filename = f"{only_image}.json"
        if os.path.exists(os.path.join(ANNOTATION_DIR, filename)):
            crop_one_image(filename)
        else:
            print(f"No annotation found: {filename}")
    else:
        json_files = sorted(
            [f for f in os.listdir(ANNOTATION_DIR) if f.endswith(".json")],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        total = 0
        for f in json_files:
            total += crop_one_image(f)

        subdirs = [d for d in os.listdir(SPLIT_DIR)
                   if os.path.isdir(os.path.join(SPLIT_DIR, d))]
        print(f"\nDone: {len(json_files)} images -> {total} spines | {len(subdirs)} subdirs in {SPLIT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-crop book spines from annotations")
    parser.add_argument("--image", type=int, default=None, help="Only crop image N")
    args = parser.parse_args()
    crop_all(args.image)
