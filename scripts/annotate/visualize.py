"""
将 Labelme 标注 JSON 渲染到原图上，生成可视化标注图。

每本书脊多边形用随机颜色半透明填充 + 白色边框绘制，保存到 data/vis/。

用法:
    python -m scripts.annotate.visualize
    python -m scripts.annotate.visualize --photo_id 1
    python -m scripts.annotate.visualize --start 1 --end 10
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from scripts.utils.paths import RAW_DIR, ANNOTATIONS_DIR, VIS_DIR

ALPHA = 0.35
BORDER_THICKNESS = 2


def random_color():
    return (random.randint(60, 230), random.randint(60, 230), random.randint(60, 230))


def draw_annotations(image: np.ndarray, shapes: list[dict]) -> np.ndarray:
    overlay = image.copy()
    for shape in shapes:
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        pts = np.array(points, dtype=np.int32)
        color = random_color()

        # 半透明填充
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        colored = np.full_like(overlay, color, dtype=np.uint8)
        overlay[mask == 255] = cv2.addWeighted(
            colored[mask == 255], ALPHA,
            overlay[mask == 255], 1 - ALPHA, 0,
        )

    # 边框画在上层
    result = overlay.copy()
    for shape in shapes:
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(result, [pts], isClosed=True, color=(255, 255, 255), thickness=BORDER_THICKNESS)

    return result


def visualize_one(photo_id: int) -> bool:
    img_path = None
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = RAW_DIR / f"{photo_id}{ext}"
        if candidate.exists():
            img_path = candidate
            break

    if img_path is None:
        print(f"  [skip] no image for {photo_id}")
        return False

    json_path = ANNOTATIONS_DIR / f"{photo_id}.json"
    if not json_path.exists():
        print(f"  [skip] no annotation for {photo_id}")
        return False

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [skip] cannot read {img_path}")
        return False

    data = json.loads(json_path.read_text(encoding="utf-8"))
    shapes = data.get("shapes", [])
    if not shapes:
        print(f"  [empty] {photo_id} has no shapes")
        return False

    result = draw_annotations(img, shapes)

    out_path = VIS_DIR / f"{photo_id}{img_path.suffix}"
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result)
    print(f"  {photo_id}: {len(shapes)} spines -> {out_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize annotations on raw images")
    parser.add_argument("--photo_id", type=int, help="Single photo ID")
    parser.add_argument("--start", type=int, default=1, help="Start photo ID")
    parser.add_argument("--end", type=int, default=None, help="End photo ID")
    args = parser.parse_args()

    if args.photo_id:
        visualize_one(args.photo_id)
        return

    images = sorted(
        [f for f in RAW_DIR.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda f: int(f.stem),
    )
    end_idx = args.end if args.end else len(images)
    end_idx = min(end_idx, len(images))

    done = 0
    for i in range(args.start - 1, end_idx):
        photo_id = int(images[i].stem)
        if visualize_one(photo_id):
            done += 1

    print(f"\nDone: {done}/{end_idx - args.start + 1} images -> {VIS_DIR}/")


if __name__ == "__main__":
    main()