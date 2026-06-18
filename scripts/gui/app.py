"""
图书盘点 GUI 模块（PyQt5）。

功能：
    - 上传图片 / 视频进行盘点
    - 显示书脊分割结果
    - 显示 OCR 识别书名
    - 显示馆藏匹配结果
    - 统计输出：书名 + 数量

使用方式：
    python -m scripts.gui.app
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def main():
    # TODO: 实现 PyQt5 界面
    raise NotImplementedError("请实现 PyQt5 界面")


if __name__ == "__main__":
    main()