"""
通用工具模块：路径配置、数据加载等。
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
SPLIT_DIR = DATA_DIR / "split"
OCR_RESULTS_DIR = DATA_DIR / "ocr_results"
CATALOG_DIR = DATA_DIR / "catalog"
PADDLE_OCR_RESULTS_DIR = DATA_DIR / "paddle_ocr_results"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
DATASET_DIR = OUTPUT_DIR / "dataset"