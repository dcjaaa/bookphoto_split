"""
从 YOLO 预测结果裁剪书脊

用法:
    python scripts/crop_from_yolo.py runs/segment/predict
    python scripts/crop_from_yolo.py runs/segment/predict --output data/split_yolo
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.paths import PROJECT_ROOT


def crop_spines_from_predict(predict_dir: str, output_dir: str | None = None):
    predict_dir = Path(predict_dir)
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "split_yolo"
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

        h, w = img.shape[:2]
        lines = label_path.read_text().strip().splitlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 7:
                continue

            coords = [float(x) for x in parts[1:]]
            points = [(int(coords[j] * w), int(coords[j + 1] * h)) for j in range(0, len(coords), 2)]
            if not points:
                continue

            xs, ys = zip(*points)
            x1, y1 = max(0, min(xs)), max(0, min(ys))
            x2, y2 = min(w, max(xs)), min(h, max(ys))

            crop = img[y1:y2, x1:x2]
            out_path = output_dir / f"{basename}_spine_{i:03d}.jpg"
            cv2.imwrite(str(out_path), crop)
            total_crops += 1

        print(f"{basename}: cropped {len(lines)} spines")

    print(f"\nDone: {total_crops} spine images -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop spines from YOLO prediction results")
    parser.add_argument("predict_dir", help="YOLO predict output directory")
    parser.add_argument("--output", default=None, help="Output directory (default: data/split_yolo)")
    args = parser.parse_args()
    crop_spines_from_predict(args.predict_dir, args.output)