"""
一次标注 → 两路训练数据

输入: Labelme JSON标注 + 原始图片
输出:
  dataset/
    images/
      train/    原图 + 裁剪单本书图
      val/
    labels/
      train/    原图YOLO标签 + 裁剪单本书YOLO标签
      val/

用法:
    python labelme_to_yolo_v2.py ./book_images ./book_labels ./dataset
"""
import os
import json
import glob
import random
import shutil
import numpy as np
import cv2


def create_yolo_line(class_id, points, img_w, img_h):
    """将多边形顶点转YOLO归一化格式"""
    norm = []
    for x, y in points:
        norm.append(f"{x / img_w:.6f}")
        norm.append(f"{y / img_h:.6f}")
    return f"{class_id} " + " ".join(norm)


def crop_spine_and_make_label(img, points, spine_idx, basename, out_img_dir, out_label_dir, class_id=0):
    """从原图裁剪一本书脊，并生成对应的YOLO标签"""
    h, w = img.shape[:2]
    pts = np.array(points, dtype=np.float32)

    # 边界框
    x1 = max(0, int(pts[:, 0].min()))
    y1 = max(0, int(pts[:, 1].min()))
    x2 = min(w, int(pts[:, 0].max()))
    y2 = min(h, int(pts[:, 1].max()))

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return

    crop_h, crop_w = crop.shape[:2]

    # 保存裁剪图片
    img_name = f"{basename}_crop_{spine_idx:03d}.jpg"
    img_path = os.path.join(out_img_dir, img_name)
    cv2.imwrite(img_path, crop)

    # 多边形坐标转换到裁剪图坐标系
    new_points = []
    for px, py in points:
        new_points.append((px - x1, py - y1))

    # 生成YOLO标签（在裁剪图中归一化）
    yolo_line = create_yolo_line(class_id, new_points, crop_w, crop_h)
    label_path = os.path.join(out_label_dir, f"{basename}_crop_{spine_idx:03d}.txt")
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(yolo_line + "\n")


def build_dataset(image_dir, json_dir, output_dir, split_ratio=0.8, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not json_files:
        print("No JSON annotation files found")
        return

    random.seed(seed)
    random.shuffle(json_files)
    split_idx = int(len(json_files) * split_ratio)
    splits = {"train": json_files[:split_idx], "val": json_files[split_idx:]}

    img_exts = [".jpg", ".jpeg", ".png"]
    stats = {"original": 0, "crops": 0}

    for split_name, files in splits.items():
        img_dir_out = os.path.join(output_dir, "images", split_name)
        label_dir_out = os.path.join(output_dir, "labels", split_name)

        for json_path in files:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            basename = os.path.splitext(os.path.basename(json_path))[0]
            img_w = data.get("imageWidth", 0)
            img_h = data.get("imageHeight", 0)

            # === 路1: 原图 + 原始标注 ===
            src_img = None
            for ext in img_exts:
                candidate = os.path.join(image_dir, basename + ext)
                if os.path.exists(candidate):
                    src_img = candidate
                    break

            if src_img is None:
                # JSON内嵌图片
                img_data = data.get("imageData")
                if img_data:
                    import base64
                    img_bytes = base64.b64decode(img_data)
                    src_img = os.path.join(image_dir, basename + ".jpg")
                    with open(src_img, "wb") as f:
                        f.write(img_bytes)
                else:
                    print(f"[skip] no image: {basename}")
                    continue

            shutil.copy2(src_img, os.path.join(img_dir_out, os.path.basename(src_img)))

            # 原图YOLO标签
            lines = []
            for shape in data["shapes"]:
                lines.append(create_yolo_line(0, shape["points"], img_w, img_h))

            label_out = os.path.join(label_dir_out, basename + ".txt")
            with open(label_out, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            stats["original"] += 1

            # === 路2: 裁剪单书本 + 对应标签 ===
            img = cv2.imread(src_img)
            if img is None:
                continue
            for i, shape in enumerate(data["shapes"]):
                crop_spine_and_make_label(
                    img, shape["points"], i, basename,
                    img_dir_out, label_dir_out, class_id=0
                )
                stats["crops"] += 1

    # 生成 data.yaml
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"""# YOLO train config
path: {os.path.abspath(output_dir)}
train: {os.path.abspath(os.path.join(output_dir, 'images', 'train'))}
val: {os.path.abspath(os.path.join(output_dir, 'images', 'val'))}

nc: 1
names:
  0: book
""")

    print(f"\n===== Done =====")
    print(f"Original images: {stats['original']}")
    print(f"Cropped spines:  {stats['crops']}")
    print(f"Total images:    {stats['original'] + stats['crops']}")
    print(f"Train/Val split: {int(split_ratio*100)}/{int((1-split_ratio)*100)}")
    print(f"Output:          {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="One annotation -> dual training data")
    parser.add_argument("image_dir", help="Original images dir")
    parser.add_argument("json_dir", help="Labelme JSON dir")
    parser.add_argument("output_dir", nargs="?", default="./dataset", help="Output dir")
    parser.add_argument("--split", type=float, default=0.8, help="Train ratio")
    args = parser.parse_args()

    build_dataset(args.image_dir, args.json_dir, args.output_dir, args.split)
