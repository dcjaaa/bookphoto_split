"""
Labelme JSON → YOLO Segmentation 完整训练数据集

目录结构:
    dataset/
    ├── images/
    │   ├── train/          ← 训练图片
    │   └── val/            ← 验证图片
    ├── labels/
    │   ├── train/          ← YOLO标签 (txt)
    │   └── val/
    ├── data.yaml           ← 训练配置
    └── titles.txt          ← 书名目录 (供OCR纠错用)

标签格式:
    <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
    坐标归一化到 [0, 1]

用法:
    python labelme_json_to_yolo.py ./book_images ./book_labels ./dataset
    python labelme_json_to_yolo.py ./book_images ./book_labels ./dataset --titles
"""
import os
import json
import glob
import random
import shutil
import argparse


def convert_labelme_to_yolo(json_dir, image_dir, output_dir, class_map=None, split_ratio=0.8, seed=42):
    if class_map is None:
        class_map = {"book": 0}

    # 创建目录
    for sub in ["images/train", "images/val", "labels/train", "labels/val", "titles"]:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not json_files:
        print("未找到 JSON 标注文件")
        return

    # 图片扩展名列表
    img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

    # 打乱并划分
    random.seed(seed)
    random.shuffle(json_files)
    split_idx = int(len(json_files) * split_ratio)
    splits = {"train": json_files[:split_idx], "val": json_files[split_idx:]}

    stats = {"total": len(json_files), "train": 0, "val": 0, "polygons": 0}

    for split_name, files in splits.items():
        for json_path in files:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            img_w = data.get("imageWidth")
            img_h = data.get("imageHeight")
            if not img_w or not img_h:
                print(f"[跳过] {os.path.basename(json_path)}: 缺少图片尺寸")
                continue

            # 生成 YOLO 标签
            lines = []
            for shape in data.get("shapes", []):
                label = shape.get("label", "")
                if label not in class_map:
                    continue
                class_id = class_map[label]
                points = shape.get("points", [])
                if not points:
                    continue

                normalized = []
                for x, y in points:
                    normalized.append(f"{x / img_w:.6f}")
                    normalized.append(f"{y / img_h:.6f}")

                lines.append(f"{class_id} " + " ".join(normalized))
                stats["polygons"] += 1

            if not lines:
                continue

            basename = os.path.splitext(os.path.basename(json_path))[0]

            # 写入标签文件
            txt_path = os.path.join(output_dir, "labels", split_name, f"{basename}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            # 拷贝对应图片
            image_path = os.path.join(image_dir, basename + ".*")
            copied = False
            for ext in img_exts:
                src = os.path.join(image_dir, basename + ext)
                if os.path.exists(src):
                    dst = os.path.join(output_dir, "images", split_name, basename + ext)
                    shutil.copy2(src, dst)
                    copied = True
                    break
            if not copied:
                # 尝试图片在原标注文件中以其他方式存储
                img_data = data.get("imageData")
                if img_data:
                    import base64
                    img_bytes = base64.b64decode(img_data)
                    # 默认存为jpg
                    dst = os.path.join(output_dir, "images", split_name, basename + ".jpg")
                    with open(dst, "wb") as f:
                        f.write(img_bytes)
                    copied = True

            if copied:
                stats[split_name] += 1
            else:
                print(f"[警告] 找不到图片: {basename}.*")

    # 生成 data.yaml
    names = {v: k for k, v in class_map.items()}
    names_str = "\n".join(f"  {v}: {names[v]}" for v in sorted(names.keys()))

    yaml_path = os.path.join(output_dir, "data.yaml")
    train_img = os.path.abspath(os.path.join(output_dir, "images", "train"))
    val_img = os.path.abspath(os.path.join(output_dir, "images", "val"))

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"""# YOLO 书脊检测训练配置
path: {os.path.abspath(output_dir)}
train: {train_img}
val: {val_img}

nc: {len(class_map)}
names:
{names_str}
""")

    print(f"\n===== 数据集准备完成 =====")
    print(f"图片总数: {stats['total']} → 训练 {stats['train']} | 验证 {stats['val']}")
    print(f"书脊标注总数: {stats['polygons']}")
    print(f"输出目录: {output_dir}")
    print(f"训练配置: {yaml_path}")

    return stats


def extract_titles(json_dir, output_dir):
    """提取书名 → 生成馆藏目录（供OCR纠错用）"""
    titles = {}
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))

    # 方式1: shape.flags.title
    # 方式2: shape.description  
    # 方式3: 标签名 "book_书名"
    for json_path in json_files:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        for shape in data.get("shapes", []):
            label = shape.get("label", "")
            title = ""

            # 尝试从三种方式提取书名
            flags = shape.get("flags", {})
            if isinstance(flags, dict) and "title" in flags:
                title = flags["title"]
            elif shape.get("description"):
                title = shape["description"]
            elif label.startswith("book_") and len(label) > 5:
                title = label[5:]  # "book_算法导论" → "算法导论"

            if title:
                titles[title] = titles.get(title, 0) + 1

    if not titles:
        print("未提取到书名（请在Labelme中用以下方式标注书名）：")
        print("  方式1: 标签名写 'book_算法导论'")
        print("  方式2: 右键shape → Edit Label → Description 填写书名")
        return

    # 保存馆藏目录
    titles_path = os.path.join(output_dir, "titles.txt")
    with open(titles_path, "w", encoding="utf-8") as f:
        for title, count in sorted(titles.items(), key=lambda x: -x[1]):
            f.write(f"{title}\n")

    print(f"\n===== 书名目录 =====")
    for title, count in sorted(titles.items(), key=lambda x: -x[1])[:20]:
        print(f"  {title} × {count}")
    if len(titles) > 20:
        print(f"  ... 共 {len(titles)} 种")
    print(f"已保存至: {titles_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Labelme JSON → YOLO 完整训练数据集")
    parser.add_argument("image_dir", help="原始图片目录")
    parser.add_argument("json_dir", help="Labelme JSON标注目录")
    parser.add_argument("output_dir", nargs="?", default="./dataset", help="YOLO数据集输出目录")
    parser.add_argument("--titles", action="store_true", help="提取书名并生成馆藏目录")
    parser.add_argument("--split", type=float, default=0.8, help="训练集比例 (默认0.8)")
    args = parser.parse_args()

    convert_labelme_to_yolo(args.json_dir, args.image_dir, args.output_dir, split_ratio=args.split)

    if args.titles:
        extract_titles(args.json_dir, args.output_dir)
