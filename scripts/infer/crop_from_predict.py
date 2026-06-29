"""
从 YOLO 预测结果裁剪书脊

用法:
    python -m scripts.infer.crop_from_predict runs/segment/predict
    python -m scripts.infer.crop_from_predict runs/segment/predict --output data/crops_yolo
"""

import argparse
from pathlib import Path

import cv2

from scripts.utils.paths import CROPS_PREDICT_DIR
from scripts.utils.crop import crop_bbox_only, polygon_bbox


def crop_spines_from_predict(predict_dir: str, output_dir: str | None = None):
    predict_dir = Path(predict_dir)
    if output_dir is None:
        output_dir = str(CROPS_PREDICT_DIR)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_dir = predict_dir / "labels"
    label_files = sorted(label_dir.glob("*.txt"))
    if not label_files:
        print("No prediction label files found")
        return

    total_crops = 0
    for label_path in label_files:
        basename = label_path.stem
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = predict_dir / f"{basename}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            print(f"[skip] no image: {basename}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[skip] cannot read: {img_path}")
            continue

        lines = label_path.read_text().strip().splitlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 7:
                continue

            coords = [float(x) for x in parts[1:]]
            points = [(int(coords[j] * len(img[0])), int(coords[j + 1] * len(img))) for j in range(0, len(coords), 2)]
            if not points:
                continue

            bbox = polygon_bbox(points)
            crop = crop_bbox_only(img, bbox)
            if crop is None:
                continue
            out_path = output_dir / f"{basename}_spine_{i:03d}.jpg"
            cv2.imwrite(str(out_path), crop)
            total_crops += 1

        print(f"{basename}: cropped {len(lines)} spines")

    print(f"\nDone: {total_crops} spine images -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop spines from YOLO prediction results")
    parser.add_argument("predict_dir", help="YOLO predict output directory")
    parser.add_argument("--output", default=None, help="Output directory (default: data/crops_yolo)")
    args = parser.parse_args()
    crop_spines_from_predict(args.predict_dir, args.output)
