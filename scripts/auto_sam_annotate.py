"""
全自动书脊标注：SAM3 根据文本提示直接检测+分割

原理:
    SAM3 一步完成：texts=["book spine"] → 检测框 + 精确多边形
    不需要分两步（YoloWorld → SAM），SAM3 内置语言编码器。

用法:
    python scripts/auto_sam_annotate.py                    # 全部294张
    python scripts/auto_sam_annotate.py --start 1 --end 5  # 先试5张
    python scripts/auto_sam_annotate.py --start 1 --end 5 --prompt "book" --score 0.3
"""
import os
import sys
import json
import cv2
import numpy as np
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from osam.apis import generate
from osam.types._generate import GenerateRequest
from osam.types._prompt import Prompt

RAW_DIR = "data/raw"
ANNOTATION_DIR = "data/annotations"

# 书脊形状过滤
MIN_ASPECT_RATIO = 1.5
MIN_HEIGHT = 80
MIN_AREA_RATIO = 0.003


def mask_to_polygon(mask, simplify_epsilon=2.0):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = simplify_epsilon / 1000.0 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = approx.squeeze(axis=1).tolist()
    if isinstance(points[0], int):
        return None
    if len(points) < 4:
        return None
    return points


def is_spine_shape(bb, img_h, img_w):
    w_box = bb.xmax - bb.xmin
    h_box = bb.ymax - bb.ymin
    if w_box <= 0 or h_box <= 0 or h_box < MIN_HEIGHT:
        return False
    ratio = max(h_box, w_box) / max(min(h_box, w_box), 1)
    if ratio < MIN_ASPECT_RATIO:
        return False
    if (w_box * h_box) / (img_w * img_h) < MIN_AREA_RATIO:
        return False
    return True


def annotate_one(image_path, prompt_text="book spine", score_threshold=0.2):
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [skip] cannot read {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # SAM3 一步完成检测+分割
    request = GenerateRequest(
        model="sam3",
        image=img_rgb,
        prompt=Prompt(
            texts=[prompt_text],
            score_threshold=score_threshold,
            max_annotations=100,
        ),
    )
    response = generate(request)

    shapes = []
    raw_count = len(response.annotations)
    for ann in response.annotations:
        if not is_spine_shape(ann.bounding_box, h, w):
            continue
        polygon = mask_to_polygon(ann.mask)
        if polygon is None:
            continue
        shapes.append({
            "label": "book",
            "points": polygon,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {},
        })

    print(f"  SAM3: {raw_count} raw -> {len(shapes)} spines")

    return {
        "version": "6.3.1",
        "flags": {},
        "shapes": shapes,
        "imageHeight": h,
        "imageWidth": w,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
    }


def run_batch(start_idx=1, end_idx=None, prompt="book spine", score=0.2):
    os.makedirs(ANNOTATION_DIR, exist_ok=True)

    images = sorted(
        [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    if not images:
        print(f"No images in {RAW_DIR}/")
        return

    if end_idx is None:
        end_idx = len(images)
    end_idx = min(end_idx, len(images))
    start_idx = max(1, start_idx)

    total_spines = 0
    print(f"Range: {start_idx}-{end_idx} / {len(images)} images")
    print(f"Model: sam3 | prompt='{prompt}' | score>{score}")
    print()

    for i in range(start_idx - 1, end_idx):
        filename = images[i]
        name = os.path.splitext(filename)[0]
        img_path = os.path.join(RAW_DIR, filename)
        json_path = os.path.join(ANNOTATION_DIR, f"{name}.json")

        print(f"[{i+1}/{end_idx}] {filename} ...", end="")
        result = annotate_one(img_path, prompt, score)
        if result is None:
            print(" FAILED")
            continue

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        n = len(result["shapes"])
        total_spines += n
        print(f" OK ({n} spines)")

    print(f"\nDone: {end_idx-start_idx+1} images -> {total_spines} book spines")
    print(f"Output: {ANNOTATION_DIR}/")
    print(f"\nNext: python scripts/auto_crop_annotations.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--prompt", type=str, default="book spine")
    parser.add_argument("--score", type=float, default=0.2)
    args = parser.parse_args()

    run_batch(args.start, args.end, args.prompt, args.score)
