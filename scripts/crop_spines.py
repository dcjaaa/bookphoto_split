"""
根据 YOLO 分割预测结果，将每个书脊单独裁剪成图片

用法:
    python crop_spines.py ../runs/segment/predict ../spines_output
"""
import os
import cv2
import glob
import argparse


def crop_spines_from_predict(predict_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    label_dir = os.path.join(predict_dir, "labels")
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))

    if not label_files:
        print("未找到预测标签文件")
        return

    total_crops = 0
    for label_path in label_files:
        basename = os.path.splitext(os.path.basename(label_path))[0]

        # 找对应原图（支持多种扩展名）
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = os.path.join(predict_dir, basename + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            print(f"[跳过] 找不到图片: {basename}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[跳过] 无法读取: {img_path}")
            continue

        h, w = img.shape[:2]

        with open(label_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 7:  # 至少3个顶点(6个值)
                continue

            class_id = parts[0]
            coords = [float(x) for x in parts[1:]]

            # 从归一化坐标还原为像素坐标
            points = []
            for j in range(0, len(coords), 2):
                if j + 1 >= len(coords):
                    break
                px = int(coords[j] * w)
                py = int(coords[j + 1] * h)
                points.append((px, py))

            if not points:
                continue

            # 计算边界框
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1, y1 = max(0, min(xs)), max(0, min(ys))
            x2, y2 = min(w, max(xs)), min(h, max(ys))

            # 裁剪并保存
            crop = img[y1:y2, x1:x2]
            out_name = f"{basename}_spine_{i:03d}.jpg"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, crop)
            total_crops += 1

        print(f"{basename}: 裁剪 {len(lines)} 本书")

    print(f"\n全部完成，共 {total_crops} 张书脊图片 → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从YOLO预测结果裁剪书脊")
    parser.add_argument("predict_dir", help="YOLO predict输出目录")
    parser.add_argument("output_dir", nargs="?", default="./spines_output", help="裁剪输出目录")
    args = parser.parse_args()

    crop_spines_from_predict(args.predict_dir, args.output_dir)
