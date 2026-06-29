# AGENTS.md

给 OpenCode agent 的工作备忘。完整说明见 `README.md`（中文），本文件只记容易踩错的地方。

## 工具链

- 依赖管理用 **uv**，不是 pip。`pyproject.toml` + `uv.lock` 是唯一来源。
  - 完整后端环境：`uv sync`
  - 仅前端（PyQt5 + httpx，不含 torch）：`uv sync --extra frontend`
  - 清华 PyPI 镜像已在 `~/.config/uv/uv.toml` 全局配置。
- 主 venv 为 Python 3.14（`.python-version`）；`pyproject.toml` 只要求 `>=3.10`。
- **没有配置任何测试 / lint / typecheck / formatter / CI。** 不要臆造 `pytest`/`ruff`/`mypy` 命令——验证方式是直接跑对应脚本。无 pre-commit hook。

## 项目结构

```
scripts/
├── annotate/   # 标注：sam / labelme / visualize
├── prepare/    # 数据准备：crop_spines / to_yolo
├── infer/      # 推理：predict / crop_from_predict
├── ocr/        # OCR：qwen_pipeline(主) / paddle_pipeline + paddle_cli(备选)
├── match/      # 馆藏匹配+计数：inventory / clean_catalog
├── api/        # FastAPI 后端
├── gui/        # PyQt5 前端
└── utils/      # paths.py(唯一路径来源) / crop.py(公共裁剪)
```

所有脚本统一用 `python -m scripts.<包>.<模块>` 调用，内部 `from scripts.utils.paths import ...`。不再有 `sys.path.insert` hack 或 `from utils.paths` 短路径。

## 路径管理

- **所有路径集中在 `scripts/utils/paths.py`**，其他文件只导入不定义。改路径只改这一处。
- `SEG_MODEL_PATH` 只在 `paths.py` 定义一处（`server.py` 也从这里导入，不再重复）。
- 裁剪逻辑集中在 `scripts/utils/crop.py`（`crop_with_mask` / `crop_bbox_only` / `polygon_to_mask`），4 处调用统一。

## 启动方式

- 后端：`uvicorn scripts.api.server:app --reload --host 0.0.0.0 --port 8000`（或 `python -m scripts.api.server`）。API 在 http://localhost:8000，文档 `/docs`。
- 前端：`python -m scripts.gui.app`。GUI **自动拉起并停止 FastAPI 后端**子进程（通过 `/proc` 杀掉占用 8000 端口的旧进程）。用 GUI 时不要再单独起 uvicorn。

## 关键陷阱

- **训练与推理 `imgsz` 必须一致用 960**，不可混用。尺寸失配会掉点：960训→640推 掉 ~0.035（0.844→0.809），640训→1280推 崩到 ~0.29（特征尺度失配）。API 默认已是 960（`scripts/api/server.py`），不要调低。
- **模型路径**在 `scripts/utils/paths.py` 的 `SEG_MODEL_PATH`，指向 `runs/segment/output/runs/book_spine_seg-26s-960/weights/best.pt`。换部署 run 只改这一处。权重缺失 → API 返回 503。
- **OCR 依赖 `.env`**：`cp .env.example .env` 并填 `SILICONFLOW_API_KEY`。`scripts/ocr/qwen_pipeline.py` 从项目根加载 `.env`。OCR 走网络，单张约 30–180s。
- PaddleOCR 跑在**独立 `.venv-paddle`**（Python 3.13），因 `paddlepaddle` 与主环境冲突。主环境通过 subprocess 调用：`.venv-paddle/bin/python scripts/ocr/paddle_cli.py`。不要在主 venv 里 import paddle。
- 馆藏匹配的两条路径读的目录不同：API `/api/inventory*` 读 `titles.json`（未清洗），CLI `scripts.match.inventory` 优先读 `titles_cleaned.json`（回退 `titles.json`）。可能给出不同匹配结果。

## 数据与权重（均不入库）

- `data/`、`predictions/`、`runs/`、`output/dataset/`、`output/gui/`、`.env`、`logs/` 及 `*.pt` 权重都不在仓库里——这是纯脚本仓库。
- 数据目录命名：`data/vis/`（标注可视化）、`data/crops_labelme/`（标注裁剪）、`predictions/crops/`（预测裁剪）、`logs/`（训练日志）。
- 克隆后需创建目录：`mkdir -p data/raw data/catalog runs/segment/output/runs/book_spine_seg-26s-960/weights logs`
- 端到端跑通需要自备：`data/raw/` 原图、`SEG_MODEL_PATH` 指向的 `best.pt`、`data/catalog/titles.json`、以及 `.env`。`/api/health` 会回报 model / catalog / ocr 是否就绪。
- 脚本自动创建的目录（无需手动建）：`data/annotations/`、`data/vis/`、`data/crops_labelme/`、`output/dataset/`、`predictions/`、`data/ocr_results/`、`output/gui/`。

## GUI 功能

- 4 个 tab：🔍 检测结果 / ✂️ 裁剪书脊 / 📖 OCR 结果 / 📚 匹配结果。
- 分割后预览叠加框+掩码半透明填充（`/api/segment` 返回 `polygon` 字段）。
- 裁剪书脊 tab：master-detail 布局，左侧列表点选、右侧大图预览，可单独/批量保存透明 PNG。
- 置信度滑块实时筛选（预览框/表格/裁剪缩略图三处同步），不重新调 API。
- 保存产物到 `output/gui/<时间戳>/`：`segmentation_vis.png` + `crops/*.png` + `detection.json` + `ocr_results.json` + `match_results.json`。

## 流程顺序

SAM 标注 → `scripts.prepare.to_yolo` → 训练(imgsz=960) → `scripts.infer.predict` → `scripts.ocr.qwen_pipeline` → `scripts.match.clean_catalog` → `scripts.match.inventory`。先跑 `clean_catalog` 生成 `titles_cleaned.json`，`inventory` 优先读它、否则回退 `titles.json`。详细参数见 README。
