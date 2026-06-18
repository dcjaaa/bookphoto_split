# BookPhoto Split — 基于视觉的图书盘点系统

基于 **SAM3** + **YOLO11** + **多模态API OCR** + **馆藏匹配** 的图书盘点全流程工具链，实现书脊检测、书名识别、去重计数。

---

## 项目现状（2026-06）

| 模块 | 状态 | 说明 |
|------|------|------|
| 书脊分割标注 (SAM3) | ✅ 完成 | 301 张照片已标注 |
| YOLO 分割模型 | ✅ 完成 | YOLO11s-seg, mAP50=0.99 |
| OCR 书名识别 | ✅ 基本完成 | 290/301 张已识别 |
| 馆藏匹配 + 去重计数 | ✅ 可用 | 模糊匹配 18 万条馆藏目录 |
| FastAPI 后端 | ✅ 完成 | `scripts/api/server.py` |
| 盘点界面 | ❌ **待实现** | ← 前端同学负责 |

---

## 目录结构

```
bookphoto_split/
├── scripts/
│   ├── api/                       # FastAPI 后端 ⭐
│   │   └── server.py              # HTTP 接口（分割/OCR/匹配）
│   ├── api_ocr/                   # OCR 模块
│   │   └── ocr_pipeline.py        # Qwen3-VL 多模态识别
│   ├── count/                     # 计数模块
│   │   └── inventory.py           # 馆藏匹配 + 去重计数
│   ├── ocr/                       # PaddleOCR 备选链路
│   ├── gui/                       # 界面模块 ← 前端实现
│   │   └── app.py                 # 空壳，待实现
│   ├── utils/
│   │   └── paths.py               # 统一路径配置
│   ├── sam_annotate.py            # SAM3 自动标注
│   ├── crop_spines_from_labelme.py
│   ├── labelme_to_yolo.py         # 标注 → YOLO 训练集
│   └── visualize_annotations.py   # 标注可视化
├── data/                          # 数据目录（不入库）
├── runs/                          # 训练产物（不入库）
├── requirements.txt               # 后端依赖
├── requirements-frontend.txt      # 前端依赖（轻量）
└── pyproject.toml                 # uv 项目配置
```

---

## 前端同学快速开始

### 1. 安装依赖（只需 PyQt5 + httpx）

```bash
pip install -r requirements-frontend.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 启动后端 API（需后端环境，由后端同学提供）

```bash
# 后端同学在装有 torch/ultralytics/openai 的环境运行:
uvicorn scripts.api.server:app --host 0.0.0.0 --port 8000
# 或
python -m scripts.api.server
```

### 3. 你的 PyQt5 程序调用 HTTP API

```python
import httpx

# 上传图片识别书名
with open("bookshelf.jpg", "rb") as f:
    resp = httpx.post(
        "http://localhost:8000/api/ocr",
        files={"file": ("1.jpg", f, "image/jpeg")},
        timeout=200,
    )
    books = resp.json()["books"]
```

### 4. 实现入口

编辑 `scripts/gui/app.py`（目前是空壳），运行 `python -m scripts.gui.app`。

---

## 前端需要实现的功能

前端是一个 **PyQt5 桌面盘点界面**，调用后端 HTTP API 完成图书盘点全流程。建议功能如下（具体可协商调整）：

### 必要功能

1. **图片上传/选择**
   - 选择本地书架照片（jpg/png）上传到后端
   - 支持单张/批量上传
   - 显示上传进度

2. **书脊分割可视化**
   - 调用 `POST /api/segment`，在图片上绘制检测框/掩码
   - 显示检测到的书脊数量
   - 可调置信度阈值（conf 参数）

3. **书名识别展示**
   - 调用 `POST /api/ocr`，返回 `[{book_name, count}]`
   - 以表格/列表形式展示每本书的书名和数量
   - 支持人工编辑修正（改书名、改数量）
   - 显示识别耗时和状态

4. **馆藏匹配 + 去重计数**
   - 调用 `POST /api/inventory`（传 OCR 结果）
   - 展示匹配后的统计：书名、总册数、是否匹配到馆藏
   - 区分"已匹配馆藏"和"未匹配"两类

5. **盘点结果汇总**
   - 调用 `GET /api/inventory/all` 查看全库汇总
   - 按册数排序展示 Top N
   - 支持导出结果（CSV/Excel/JSON）

### 可选功能

6. **历史结果浏览**
   - 调用 `GET /api/ocr` 列出所有已处理图片
   - 调用 `GET /api/ocr/{photo_id}` 查看单张结果

7. **馆藏目录查询**
   - 调用 `GET /api/catalog` 分页浏览馆藏书名

8. **原图与分割对比**
   - 左右对比：原图 / 标注图

### UI 布局建议

```
┌─────────────────────────────────────────────┐
│  [选择图片] [批量上传]    conf: [0.25 ▼]    │
├──────────────────────┬──────────────────────┤
│                      │  识别结果             │
│   图片预览区          │  ┌─────────────────┐ │
│  (叠加分割框)         │  │ 书名 | 数量 | 匹配│ │
│                      │  │ ...             │ │
│                      │  └─────────────────┘ │
├──────────────────────┴──────────────────────┤
│  [识别书名] [馆藏匹配] [导出]  状态: 就绪     │
└─────────────────────────────────────────────┘
```

---

## HTTP API 规格

后端运行在 `http://localhost:8000`，交互式文档：`http://localhost:8000/docs`

### 1. `POST /api/segment` — 书脊分割

上传图片 → YOLO 检测书脊位置。

**请求**：`multipart/form-data`
- `file`: 图片文件
- `conf`: 置信度阈值（默认 0.25，query 参数）
- `imgsz`: 推理尺寸（默认 1280，query 参数）

**响应**：
```json
{
  "image_size": {"width": 4080, "height": 3072},
  "count": 24,
  "boxes": [
    {"bbox": [952.0, 152.0, 1137.0, 1068.0], "confidence": 0.92, "label": "book"}
  ]
}
```

### 2. `POST /api/ocr` — 书名识别

上传图片 → Qwen3-VL 识别书名和数量（耗时 30–180s）。

**请求**：`multipart/form-data`
- `file`: 图片文件

**响应**：
```json
{
  "books": [
    {"book_name": "人品胜于能力", "count": 2},
    {"book_name": "算法导论", "count": 1}
  ],
  "count": 2
}
```

### 3. `POST /api/inventory` — 馆藏匹配

对给定 OCR 结果做模糊匹配 + 去重计数。

**请求**：`application/json`
```json
{
  "results": [
    {"photo_id": 1, "books": [{"book_name": "人品胜于能力", "count": 2}]}
  ],
  "threshold": 0.6
}
```

**响应**：
```json
{
  "book_counts": {"人品胜于能力": 2, "算法导论": 1},
  "total_copies": 3,
  "matched": 2,
  "unique_titles": 2,
  "match_log": [
    {"photo_id": 1, "ocr_name": "人品胜于能力", "matched_name": "人品胜于能力", "count": 2}
  ]
}
```

### 4. `GET /api/ocr/{photo_id}` — 取已有 OCR 结果

按图片 ID 取已保存的识别结果。

**响应**：
```json
{
  "photo_id": 1,
  "image": "1.jpg",
  "books": [{"book_name": "人品胜于能力", "count": 2}]
}
```

### 5. `GET /api/ocr` — 列出所有 OCR 结果概览

**响应**：
```json
{
  "total": 290,
  "results": [{"photo_id": 1, "book_count": 13, "has_error": false}]
}
```

### 6. `GET /api/inventory/all` — 全库盘点汇总

汇总所有已有 OCR 结果，生成盘点统计。

**响应**：
```json
{
  "book_counts": {"书名": 册数},
  "total_copies": 1234,
  "matched": 1100,
  "unique_titles": 500,
  "photos": 290
}
```

### 7. `GET /api/catalog` — 馆藏目录（分页）

**Query**：`limit=100&offset=0`

**响应**：
```json
{"total": 186053, "limit": 100, "offset": 0, "titles": ["书名1", "书名2"]}
```

### 8. `GET /api/health` — 健康检查

```json
{"status": "ok", "model_exists": true, "catalog_exists": true, "ocr_results": 290}
```

---

## 后端环境搭建（后端同学）

### 1. 安装依赖

```bash
# 方式 A: uv（推荐）
uv sync

# 方式 B: pip
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 下载模型权重（不入库）

```bash
# YOLO 预训练权重
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt

# 训练好的 best.pt 需另行获取，放入:
# runs/segment/output/runs/book_spine_seg-2/weights/best.pt
```

### 3. 配置 OCR API

```bash
cp .env.example .env
# 编辑 .env 填入 SILICONFLOW_API_KEY
```

### 4. 启动后端

```bash
uvicorn scripts.api.server:app --reload --host 0.0.0.0 --port 8000
```

### 5. PaddleOCR 备选链路（可选）

```bash
uv venv .venv-paddle --python 3.13
.venv-paddle/bin/pip install -r paddle-requirements.txt
```

---

## 全流程使用（命令行）

### Step 1：书脊分割标注

```bash
python scripts/sam_annotate.py --start 1 --end 301
```

### Step 2：训练 YOLO 分割模型

```bash
python scripts/crop_spines_from_labelme.py
python scripts/labelme_to_yolo.py
yolo segment train model=yolo11s-seg.pt data=output/dataset/data.yaml epochs=100 batch=8
```

### Step 3：OCR 识别

```bash
python -m scripts.api_ocr.ocr_pipeline --start 1 --end 301
```

### Step 4：馆藏匹配

```bash
python -m scripts.count.inventory
```

### Step 5：启动 API + 前端

```bash
# 后端
uvicorn scripts.api.server:app --port 8000
# 前端
python -m scripts.gui.app
```

---

## 模型效果

| 指标 | mAP50 | mAP50-95 | Precision | Recall |
|------|-------|----------|-----------|--------|
| Box  | 0.992 | 0.923    | 0.979     | 0.976  |
| Mask | 0.980 | 0.821    | 0.971     | 0.968  |

数据集：301 张书架照片，7586 个标注多边形，训练 100 epochs。

---

## 依赖说明

| 文件 | 用途 | 说明 |
|------|------|------|
| `requirements.txt` | 后端环境 | 含 labelme/ultralytics/openai/fastapi 等 |
| `requirements-frontend.txt` | 前端环境 | 仅 PyQt5 + httpx，轻量 |
| `paddle-requirements.txt` | PaddleOCR | 独立 venv，备选 OCR 链路 |
| `pyproject.toml` | uv 管理 | 后端同学用 `uv sync` |

---

## 工作流总览

```
拍摄照片 → data/raw/
   ↓
SAM3 书脊分割标注 → data/annotations/
   ↓
YOLO 训练 → runs/ (best.pt)
   ↓
FastAPI 后端启动 (scripts/api/server.py)
   ↓
PyQt5 前端调 HTTP API → 盘点结果
```
