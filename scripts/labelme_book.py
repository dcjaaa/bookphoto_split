"""
Labelme SAM 辅助标注脚本

用法:
    python scripts/labelme_book.py
    python scripts/labelme_book.py --start 5

快捷键:
    Ctrl+N  手动补画多边形     Ctrl+S  保存
    D       下一张             A       上一张
    Delete  删除选中的标注
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.paths import RAW_DIR, ANNOTATIONS_DIR


def launch_labelme(start_idx: int = 1):
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(
        [f for f in RAW_DIR.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda f: int(f.stem),
    )

    if not images:
        print(f"No images found in {RAW_DIR}/")
        return

    start_idx = max(1, min(start_idx, len(images)))
    start_file = str(images[start_idx - 1])

    print(f"Total: {len(images)} images | Start from: #{start_idx}")
    print(f"Annotations -> {ANNOTATIONS_DIR}/")
    print()
    print("=== SAM workflow ===")
    print("1. AI Tools menu -> Detectors/Interactors -> SAM")
    print("2. Click on each book spine -> SAM generates polygon")
    print("3. Label = 'book' -> Ctrl+S -> D (next)")

    cmd = ["labelme", start_file, "--output", str(ANNOTATIONS_DIR)]
    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Labelme SAM annotation")
    parser.add_argument("--start", type=int, default=1, help="Start from image N")
    args = parser.parse_args()
    launch_labelme(args.start)