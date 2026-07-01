"""
项目路径唯一来源。

所有脚本只从这里导入路径常量，不在本地重复定义。
改路径只需改这一处。
"""

from pathlib import Path

# === 项目根 ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# === data/ ===
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                  # 原始书架照片
ANNOTATIONS_DIR = DATA_DIR / "annotations"  # SAM/labelme 标注 JSON
VIS_DIR = DATA_DIR / "vis"                  # 标注可视化图 (原 annotated/)
CROPS_LABELED_DIR = DATA_DIR / "crops_labelme"  # 按标注裁剪书脊 (原 split/)
OCR_RESULTS_DIR = DATA_DIR / "ocr_results"  # Qwen3-VL OCR 结果
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"  # 测评基准(匹配后的OCR结果,供人工审查)
CATALOG_DIR = DATA_DIR / "catalog"          # 馆藏目录
PADDLE_OCR_RESULTS_DIR = DATA_DIR / "paddle_ocr_results"
CROPS_PREDICT_DIR = DATA_DIR / "crops_yolo"  # 从 YOLO 预测裁剪 (原 split_yolo/)

# === catalog 文件 ===
CATALOG_FILE = CATALOG_DIR / "titles.json"
CATALOG_RAW_FILE = CATALOG_DIR / "titles1.json"  # 原始未清洗备份(供 clean_catalog.py 读取)

# === output/ ===
OUTPUT_DIR = PROJECT_ROOT / "output"
DATASET_DIR = OUTPUT_DIR / "dataset"            # YOLO 训练数据集
GUI_OUTPUT_DIR = OUTPUT_DIR / "gui"             # GUI 保存产物
INVENTORY_RESULT_FILE = OUTPUT_DIR / "inventory_result.json"

# === predictions/ ===
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
PRED_VIS_DIR = PREDICTIONS_DIR / "vis"          # 掩码+框可视化
PRED_LABELS_DIR = PREDICTIONS_DIR / "labels"    # YOLO 归一化坐标
PRED_LABELME_DIR = PREDICTIONS_DIR / "labelme"  # labelme JSON
PRED_CROPS_DIR = PREDICTIONS_DIR / "crops"      # 裁剪书脊 (原 split/)

# === runs/ 模型权重 ===
RUNS_DIR = PROJECT_ROOT / "runs"
SEG_MODEL_PATH = RUNS_DIR / "segment" / "output" / "runs" / "book_spine_seg-26s-960-v2-2" / "weights" / "best.pt"

# === logs/ ===
LOGS_DIR = PROJECT_ROOT / "logs"

# === PaddleOCR 独立环境 ===
PADDLE_VENV_PYTHON = PROJECT_ROOT / ".venv-paddle" / "bin" / "python"
PADDLE_CLI = PROJECT_ROOT / "scripts" / "ocr" / "paddle_cli.py"
