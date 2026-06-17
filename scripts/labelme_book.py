"""
Labelme SAM 辅助标注脚本

用法:
    python scripts/labelme_book.py              # 标注全部
    python scripts/labelme_book.py --start 5    # 从第5张开始

工作流:
    1. 打开图片 → AI Tools 菜单 → 选择 SAM
    2. 每本书书脊上点一下 → SAM自动生成多边形轮廓
    3. 标签选 "book" → Ctrl+S 保存 → D 下一张

快捷键:
    Ctrl+N  手动补画多边形     Ctrl+S  保存
    D       下一张             A       上一张
    Delete  删除选中的标注
"""
import os
import subprocess
import argparse

ANNOTATION_DIR = "data/annotations"
IMAGE_DIR = "data/raw"


def launch_labelme(start_idx=1):
    os.makedirs(ANNOTATION_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    images = sorted(
        [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    if not images:
        print(f"No images found in {IMAGE_DIR}/")
        return

    # 从指定位置开始
    start_idx = max(1, min(start_idx, len(images)))
    start_file = os.path.join(IMAGE_DIR, images[start_idx - 1])

    print(f"Total: {len(images)} images | Start from: #{start_idx}")
    print(f"Annotations -> {ANNOTATION_DIR}/")
    print()
    print("=== SAM workflow ===")
    print("1. AI Tools menu -> Detectors/Interactors -> SAM")
    print("2. Click on each book spine -> SAM generates polygon")
    print("3. Label = 'book' -> Ctrl+S -> D (next)")

    cmd = [
        "labelme",
        start_file,
        "--output", ANNOTATION_DIR,
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Labelme SAM annotation")
    parser.add_argument("--start", type=int, default=1, help="Start from image N")
    args = parser.parse_args()
    launch_labelme(args.start)
