# BookPhoto Split — 基于视觉的图书盘点系统

基于 **SAM3** + **YOLO26s-Seg** + **多模态API OCR** + **馆藏匹配** 的图书盘点全流程工具链，实现书脊检测、书名识别、去重计数。

---

## 项目现状（2026-07）

| 模块 | 状态 | 数据量 | 说明 |
|------|------|--------|------|
| 书脊分割标注 (SAM3) | ✅ 完成 | 517 张 | SAM3 自动标注 + labelme 人工修正 |
| YOLO 分割模型 | ✅ 完成 | 12678 标注 | YOLO26s, holdout Mask mAP50-95=0.927 |
| OCR 书名识别 | ✅ 完成 | 单书脊 32B 并发 | 整图 Qwen3-VL-32B (297张) + 单脊逐本 OCR |
| 馆藏匹配 | ✅ 完成 | 99.3% 匹配率 | jieba 关键词分桶 (48K关键词) + 暴力遍历 |
| 书名标注数据集 | ✅ 完成 | 517 文件 / 5,966 条 | book_labels, 人工核实+匹配 |
| FastAPI 后端 | ✅ 完成 | 10 并发 | 分割/单脊OCR+匹配/馆藏匹配/目录 |
| 盘点界面 | ✅ 完成 | PyQt5 | 4 tab：检测/裁剪/OCR/评估详情 |

---

## 目录结构

```
bookphoto_split/
├── scripts/                          # 全部脚本
│   ├── api/                          # FastAPI 后端 ⭐
│   │   └── server.py                 #   HTTP 接口（分割/OCR/匹配/目录）
│   ├── annotate/                     # 标注
│   │   ├── sam.py                    #   SAM3 自动标注 ⭐
│   │   ├── labelme.py                #   Labelme 手动标注
│   │   └── visualize.py              #   标注可视化渲染
│   ├── prepare/                      # 数据准备
│   │   ├── crop_spines.py            #   按标注裁剪书脊（PNG 透明）
│   │   └── to_yolo.py                #   标注 → YOLO 训练集
│   ├── infer/                        # 推理
│   │   ├── predict.py                #   批量预测+裁剪+labelme ⭐
│   │   └── crop_from_predict.py      #   从 YOLO 预测裁剪书脊
│   ├── ocr/                          # OCR
│   │   ├── qwen_pipeline.py          #   Qwen3-VL 书名识别（主：整图+单脊） ⭐
│   │   ├── paddle_pipeline.py        #   PaddleOCR 备选链路（主环境入口）
│   │   └── paddle_cli.py             #   .venv-paddle 下的 CLI
│   ├── match/                        # 馆藏匹配 + 计数
│   │   ├── inventory.py              #   模糊匹配 + 去重计数 + 评估对比 ⭐
│   │   ├── build_keyword_index.py    #   jieba 关键词索引生成
│   │   ├── create_ground_truth.py    #   测评基准生成
│   │   └── clean_catalog.py          #   馆藏目录清洗
│   ├── gui/                          # 界面 ← 前端实现
│   │   ├── app.py                    #   PyQt5 主窗口 + 交互逻辑
│   │   ├── backend.py                #   FastAPI 进程管理器
│   │   └── client.py                 #   HTTP API 封装
│   └── utils/
│       ├── paths.py                  #   统一路径配置（唯一来源）
│       └── crop.py                   #   公共裁剪函数
├── data/                             # 数据（不入库）
│   ├── raw/                          #   原始书架照片 (1.jpg ~ 517.jpg)
│   ├── annotations/                  #   SAM 标注 (labelme JSON)
│   ├── vis/                          #   标注可视化图
│   ├── crops_labelme/                #   SAM 标注裁剪的书脊
│   ├── crops_yolo/                   #   从 YOLO 预测裁剪的书脊
│   ├── ocr_results/                  #   OCR 识别结果 JSON
│   ├── book_labels/                  #   书名标注数据集 (517文件, 5966条) ⭐
│   ├── ground_truth/                 #   测评基准 (匹配后的OCR结果)
│   ├── temp_match/                   #   临时匹配测试结果
│   └── catalog/                      #   馆藏目录 (titles.json + keyword_index.json)
├── predictions/                      # YOLO 预测产物（不入库）
│   ├── vis/                          #   掩码+框可视化图 (297)
│   ├── labels/                       #   YOLO 归一化坐标 (297)
│   ├── labelme/                      #   labelme JSON 格式 (297)
│   └── crops/                        #   预测裁剪的书脊 PNG (297 目录)
├── logs/                             # 训练日志（不入库）
├── output/                           # 系统输出
│   ├── gui/                          #   GUI 产物（标注图+OCR+匹配 JSON）
│   ├── dataset/                      #   YOLO 训练数据集（不入库）
│   └── inventory_result.json         #   盘点匹配结果 ⭐
├── runs/                             # YOLO 训练产物（不入库）
│   └── segment/output/runs/book_spine_seg-26s-960-v2-2/weights/best.pt
├── .env.example                      # API 配置模板
├── pyproject.toml                    # uv 项目配置 + 依赖（唯一依赖源）
├── uv.lock                           # 依赖锁定（uv 管理）
└── paddle-requirements.txt           # PaddleOCR 独立依赖
```

---

### 模型训练历程

| 模型 | imgsz | Box mAP50-95 | Mask mAP50-95 | 显存 | 速度 | 备注 |
|------|-------|-------------|--------------|------|------|------|
| yolo11s | 640 | 0.923 | 0.821 | 3.4G | 1.5ms | 基线，已归档 |
| yolo11l | 960 | — | — | 7.0G | 4.5ms | 10 epochs 中止 |
| yolo26l | 960 | 0.943 | 0.843 | 6.1G | 5.0ms | 已归档 |
| yolo26m | 960 | 0.602 | 0.530 | — | — | 10 epochs 测试中 |
| **yolo26s v2** | **960** | **0.950** | **0.927(holdout)** | 5.7G | 3.5ms | **当前最佳** |
| yolo26s v1 | 960 | 0.943 | 0.921(holdout) | 6.9G | 3.5ms | 297标注，已归档 |
| yolo26l | 960 | 0.943 | 0.843(val) | 6.1G | 5.0ms | 已归档 |

> 最佳模型：`runs/segment/output/runs/book_spine_seg-26s-960-v2-2/weights/best.pt`（路径只在 `scripts/utils/paths.py` 的 `SEG_MODEL_PATH` 一处定义）

---

## 快速开始:目录准备

仓库只含脚本,数据和权重需自备。克隆后先创建目录结构:

### 一键创建所有目录

```bash
mkdir -p data/raw data/catalog \
         runs/segment/output/runs/book_spine_seg-26s-960-v2-2/weights \
         logs
```

### 需要自备的文件

| 文件 | 说明 | 来源 |
|------|------|------|
| `data/raw/*.jpg` | 书架照片 | 自行拍摄 |
| `data/catalog/titles.json` | 馆藏目录(书名 JSON 数组) | 自行整理 |
| `runs/.../best.pt` | 训练好的分割模型 | 另行获取 |
| `.env` | OCR API 密钥 | `cp .env.example .env` 后填入 |

### 脚本自动生成的目录(无需手动创建)

| 目录 | 生成脚本 |
|------|----------|
| `data/annotations/` | `scripts.annotate.sam` |
| `data/vis/` | `scripts.annotate.visualize` |
| `data/crops_labelme/` | `scripts.prepare.crop_spines` |
| `output/dataset/` | `scripts.prepare.to_yolo` |
| `predictions/` | `scripts.infer.predict` |
| `data/ocr_results/` | `scripts.ocr.qwen_pipeline` |
| `output/gui/` | GUI 保存时自动创建 |

---

## 前端同学快速开始

### 1. 安装依赖（只需 PyQt5 + httpx）

```bash
uv sync --extra frontend
```

### 2. 启动 GUI（自动管理后端）

```bash
python -m scripts.gui.app
```

GUI 启动后会自动拉起 FastAPI 后端进程，无需手动启动 uvicorn。
状态指示灯变绿即表示后端就绪，可直接开始盘点。

---

## 前端功能说明

前端是一个 **PyQt5 桌面盘点界面**，调用后端 HTTP API 完成图书盘点全流程。GUI 启动后自动管理后端进程。

### 使用流程

1. **选择图片** — 工具栏「📂 选择图片」打开书架照片（jpg/png）
2. **分割检测** — 点击「🔍 分割检测」，YOLO 检测书脊位置，图片上叠加彩色检测框+掩码半透明填充，右侧显示预测质量摘要和每脊明细
   - 可拖动置信度滑块（0.10–1.00）实时筛选检测框、表格、裁剪缩略图（无需重新检测）
3. **裁剪书脊** — 切换「✂️ 裁剪书脊」tab 查看每本书脊的裁剪图，左侧列表点选、右侧大图预览，可单张/批量保存
4. **OCR 识别** — 点击「📖 OCR 识别」，逐张发单脊裁剪图到 `POST /api/ocr/spine`（10 并发、32B 模型），服务器自动完成 OCR + 馆藏匹配，返回书名和匹配结果
   - 5 列结果表：序号 / 置信度 / OCR 书名 / 馆藏匹配 / 书名标注 / 结果
   - 🟢 重试失败：重试超时/错误的脊柱
   - 🟢 重试多检：重试评估为"多检"的脊柱
   - 🟢 双击某行单独重试
5. **评估对比** — 点击「📊 评估对比」，加载 `data/book_labels/{photo_id}.json` 比对准确率
   - 第一轮：脊柱 OCR 名 vs 书名标注名（原始 OCR）
   - 第二轮（兜底）：脊柱馆藏匹配名 vs 书名标注馆藏匹配名
   - 切换到「📊 评估详情」tab 查看逐条对比（7 列详细表）
6. **保存结果** — 点击「💾 保存结果」，所有产物保存到 `output/gui/<时间戳>/`

### 界面布局

```
┌──────────────────────────────────────────────────────────────┐
│ [📂 选择图片]  置信度: [====] 0.25    🟢 后端就绪            │
├────────────────────────┬─────────────────────────────────────┤
│                        │ [🔍检测] [✂️裁剪] [📖OCR] [📊评估]    │
│   图片预览区            │                                     │
│   (叠加检测框+掩码)     │  预测质量: N个书脊 | 平均置信度...    │
│   滚轮缩放              │  ┌────┬──────┬────────┬──────┬───┐ │
│                        │  │序号│置信度│OCR书名 │馆藏  │结果│ │
│                        │  └────┴──────┴────────┴──────┴───┘ │
├────────────────────────┴─────────────────────────────────────┤
│ [🔍分割] [📖OCR] [📊评估] [🔄重试失败] [🔄重试多检] [💾保存] │
│ 状态: 就绪 — 点击「选择图片」开始                             │
└──────────────────────────────────────────────────────────────┘
```

### 代码结构

```
scripts/gui/
├── app.py          # MainWindow + ImagePreviewWidget + OCR 并发池
├── backend.py      # FastAPI 进程管理器（自动启停）
└── client.py       # httpx API 封装（segment/ocr_spine/inventory/health）

output/gui/<时间戳>/   # GUI 手动保存的产物
├── segmentation_vis.png  #   分割可视化图（框+掩码+标签）
├── crops/                #   单独书脊裁剪 PNG（透明背景）
├── detection.json        #   检测结果（bbox+置信度+多边形）
├── ocr_results.json      #   OCR 识别结果
└── match_results.json    #   评估结果
```

---

## HTTP API 规格

后端运行在 `http://localhost:8000`，交互式文档：`http://localhost:8000/docs`

### 1. `POST /api/segment` — 书脊分割

上传图片 → YOLO 检测书脊位置。

**请求**：`multipart/form-data`
- `file`: 图片文件
- `conf`: 置信度阈值（默认 0.25，query 参数）
- `imgsz`: 推理尺寸（默认 **960**，与训练一致，query 参数）

**响应**：
```json
{
  "image_size": {"width": 4080, "height": 3072},
  "count": 24,
  "boxes": [
    {"bbox": [952.0, 152.0, 1137.0, 1068.0], "confidence": 0.92, "label": "book", "polygon": [[952.0, 152.0], ...]}
  ]
}
```

> ⚠️ **训练与推理 imgsz 必须一致用 960**，不可混用。尺寸失配会掉点：960训→640推 掉 ~0.035（0.844→0.809），640训→1280推 崩到 ~0.29（特征尺度失配）。

### 2. `POST /api/ocr/spine` — 单脊 OCR + 馆藏匹配

上传**单张书脊裁剪图** → 32B Qwen3-VL 识别书名 → 自动模糊匹配馆藏目录（一步完成，耗时 10–25s）。

**请求**：`multipart/form-data`
- `file`: 单张书脊裁剪图（png/jpg）

**响应**：
```json
{
  "book_name": "人品胜于能力",
  "matched_name": "人品胜于能力",
  "score": 1.0,
  "strategy": "kw_exact",
  "needs_review": false
}
```

> ⚠️ **服务端同步端点**（非 async），通过 uvicorn 线程池支持 10 路并发。GUI 使用 3 个 API key 轮选以规避限速。

### 3. `POST /api/inventory` — 馆藏匹配

对给定 OCR 结果做模糊匹配 + 去重计数。

**请求**：`application/json`
```json
{
  "results": [
    {"photo_id": 1, "books": [{"book_name": "人品胜于能力", "count": 2}]}
  ],
  "threshold": 0.7
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
    {"photo_id": 1, "ocr_name": "人品胜于能力", "matched_name": "人品胜于能力", "score": 1.0, "strategy": "exact_full", "needs_review": false, "count": 2}
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
  "total": 297,
  "results": [{"photo_id": 1, "book_count": 13, "has_error": false}]
}
```

### 6. `GET /api/inventory/all` — 全库盘点汇总

汇总所有已有 OCR 结果，生成盘点统计。

**响应**：
```json
{
  "book_counts": {"书名": 册数},
  "total_copies": 7031,
  "matched": 3232,
  "unique_titles": 2641,
  "photos": 297
}
```

### 7. `GET /api/catalog` — 馆藏目录（分页）

**Query**：`limit=100&offset=0`

**响应**：
```json
{"total": 185976, "limit": 100, "offset": 0, "titles": ["书名1", "书名2"]}
```

### 8. `GET /api/health` — 健康检查

```json
{"status": "ok", "model_exists": true, "catalog_exists": true, "ocr_results": 297}
```

---

## 后端环境搭建

### 1. 安装依赖

```bash
# uv（推荐，全局已配清华镜像源）
uv sync
```

### 2. 下载模型权重（不入库）

```bash
# YOLO 预训练权重
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo26s-seg.pt

# 训练好的 best.pt 需另行获取，放入:
# runs/segment/output/runs/book_spine_seg-26s-960-v2-2/weights/best.pt
# (路径只在 scripts/utils/paths.py 的 SEG_MODEL_PATH 一处定义)
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
uv pip install -r paddle-requirements.txt -p .venv-paddle
```

---

## 全流程使用（命令行）

### Step 1：书脊分割标注

```bash
# SAM3 全自动批量标注
python -m scripts.annotate.sam --start 1 --end 297

# 或手动标注
python -m scripts.annotate.labelme
```

输出：`data/annotations/{n}.json`

### Step 2：训练 YOLO 分割模型

```bash
# 裁剪书脊（标注验证用）
python -m scripts.prepare.crop_spines

# 生成 YOLO 训练集（原图+裁剪双路，含 holdout 测试集）
python -m scripts.prepare.to_yolo --holdout 30

# 训练
yolo segment train \
  model=yolo26s-seg.pt data=output/dataset/data.yaml \
  epochs=100 batch=6 imgsz=960 \
  cos_lr=True patience=30 close_mosaic=20 warmup_epochs=5 \
  copy_paste=0.1 cutmix=0.1 mixup=0.1 \
  flipud=0.5 scale=0.3 \
  seed=42 workers=8 \
  project=output/runs name=book_spine_seg-26s-960-v2-2
```

### Step 3：批量预测固化

用训练好的模型对全部图片推理，保存可视化/坐标/裁剪结果。

```bash
python -m scripts.infer.predict                    # 全部
python -m scripts.infer.predict --start 1 --end 10  # 范围
python -m scripts.infer.predict --photo_id 5        # 单张
python -m scripts.infer.predict --conf 0.3          # 调阈值
```

输出：`predictions/`
- `vis/{id}.jpg` — 掩码+框可视化
- `labels/{id}.txt` — YOLO 归一化坐标
- `labelme/{id}.json` — labelme 多边形格式
- `crops/{id}/spine_{n}.png` — 裁剪书脊（透明 PNG）

> ⚠️ **训练与推理 imgsz 必须一致用 960**，不可混用。尺寸失配会掉点：960训→640推 掉 ~0.035（0.844→0.809），640训→1280推 崩到 ~0.29（特征尺度失配）。

### Step 4：OCR 识别书名

```bash
python -m scripts.ocr.qwen_pipeline --start 1 --end 297
python -m scripts.ocr.qwen_pipeline --photo_id 5      # 单张
python -m scripts.ocr.qwen_pipeline --retry-failed    # 重试空结果
```

输出：`data/ocr_results/{n}.json`

### Step 5：馆藏目录清洗 + 匹配

```bash
# 5.1 清洗馆藏目录（HTML实体反转义、去重）
python -m scripts.match.clean_catalog

# 5.2 模糊匹配 + 去重计数
python -m scripts.match.inventory
python -m scripts.match.inventory --threshold 0.7   # 调阈值
```

输出：`output/inventory_result.json`

匹配策略（优先级从高到低）：

| 策略 | 说明 | 示例 |
|------|------|------|
| exact_full | OCR 全名精确匹配馆藏 | `应用文写作` = `应用文写作` |
| exact_main | 主标题精确匹配 | `刑法各罪论 上册 修订五版` → `刑法各罪论` |
| prefix_main | 馆藏以主标题开头（后缀≤8字） | `考研英语（二）...` → `考研英语二` |
| contained | 馆藏包含于主标题 | `美学基础（第3版）` → `美学基础` |
| contained_in | 主标题包含于馆藏 | `涉外文书写作大全` → `新编涉外文书写作大全` |
| partial_full | OCR 全名 partial_ratio | 模糊兜底 |
| token_set | token_set_ratio | 空格/词序差异兜底 |
| partial_main | 主标题 partial_ratio | 模糊兜底 |
| ratio | OCR 全名 ratio | 候选集内兜底 |
| global_ratio | 全库 ratio | OCR 首字符错误兜底 |

主标题提取：按 `（ ( : ： — ——` 和空格（仅中文）切分，去版本号/册次/编者等噪声。

### Step 6：启动 API + 前端

```bash
# 后端
uvicorn scripts.api.server:app --port 8000
# 前端
python -m scripts.gui.app
```

---

## 模型效果

### 训练指标（yolo26s-seg, best.pt @ epoch 97）

| 指标 | mAP50 | mAP50-95 | Precision | Recall |
|------|-------|----------|-----------|--------|
| Box  | 0.991 | 0.933    | 0.947     | 0.973  |
| Mask | 0.991 | 0.909    | 0.945     | 0.980  |

数据集：517 张书架照片，12,678 个标注多边形，13,195 张训练图像（原图+掩码crop 双路），训练 100 epochs，显存峰值 5.68 GB。
最终模型：`runs/segment/output/runs/book_spine_seg-26s-960-v2-2/weights/best.pt`（路径只在 `scripts/utils/paths.py` 的 `SEG_MODEL_PATH` 一处定义）

### holdout 评估（30 张独立测试原图）

| 指标 | mAP50 | mAP50-95 | Precision | Recall |
|------|-------|----------|-----------|--------|
| Box  | 0.993 | 0.950    | 0.977     | 0.985  |
| Mask | 0.993 | 0.927    | 0.976     | 0.988  |

holdout 测试集为 30 张从未参与训练/验证的原图（744 张含 crop 图像，1,427 个实例），反映真实部署效果。

训练报告详见 `docs/training_notes.md`。

### 馆藏匹配效果

匹配引擎采用 **jieba 关键词分桶 + 暴力遍历** 策略，替代了早期的多级 fuzzy_match 策略。

| 指标 | 值 |
|------|-----|
| 馆藏目录 | 185,976 条 |
| jieba 关键词 | 48,658 个 |
| 关键词索引大小 | 6.1 MB |
| 书名标注条目 | 5,966 条 |
| 匹配率（temp_match 全量） | **99.3%** |
| 匹配耗时 | 单张 <0.3s（关键词桶） / 全量 40s |

---

## 脚本功能速查

| 脚本（`python -m ...`） | 功能 |
|------|------|
| `scripts.annotate.sam` | SAM3 全自动批量标注 ⭐ |
| `scripts.annotate.labelme` | 启动 Labelme 手动标注 |
| `scripts.annotate.visualize` | 标注可视化渲染 |
| `scripts.prepare.crop_spines` | 按标注裁剪书脊（PNG 透明） |
| `scripts.prepare.to_yolo` | 标注 → YOLO seg 训练集（掩码crop双路, --no-crop 纯全图） |
| `scripts.infer.predict` | 批量预测+裁剪+labelme ⭐ |
| `scripts.infer.crop_from_predict` | 从 YOLO 预测结果裁剪书脊 |
| `scripts.ocr.qwen_pipeline` | Qwen3-VL 多模态书名识别 ⭐ |
| `scripts.ocr.paddle_pipeline` | PaddleOCR 备选链路 |
| `scripts.match.inventory` | 馆藏匹配 + 去重计数 + 评估对比 ⭐ |
| `scripts.match.build_keyword_index` | jieba 关键词索引生成 |
| `scripts.match.create_ground_truth` | 测评基准生成 |
| `scripts.match.clean_catalog` | 馆藏目录清洗 |
| `scripts.api.server` | FastAPI HTTP 后端 ⭐ |
| `scripts.gui.app` | PyQt5 盘点界面 ✅ |

---

## 依赖说明

项目完全用 **uv** 管理依赖，pyproject.toml 为唯一依赖源，uv.lock 锁定版本。全局已配清华镜像源（`~/.config/uv/uv.toml`）。

| 命令 | 用途 | 说明 |
|------|------|------|
| `uv sync` | 后端环境 | 含 labelme/ultralytics/openai/fastapi/rapidfuzz 等 |
| `uv sync --extra frontend` | 前端环境 | 仅 PyQt5 + httpx，轻量 |
| `uv pip install -r paddle-requirements.txt -p .venv-paddle` | PaddleOCR | 独立 venv，备选 OCR 链路 |

主要依赖：
- **jieba** — 中文分词，馆藏关键词索引
- **labelme** + **osam** — 标注平台 + SAM3 自动化
- **ultralytics** — YOLO 训练与推理
- **openai** — Qwen3-VL 多模态 OCR
- **rapidfuzz** — 模糊匹配（比标准库快 100x）
- **fastapi** + **uvicorn** — HTTP 后端
- **PyQt5** — 盘点界面
- **opencv-python-headless** — 图像处理

---

## 工作流总览

```
拍摄照片 → data/raw/
   ↓
SAM3 书脊分割标注 → data/annotations/
   ↓
YOLO 训练 (imgsz=960) → runs/ (best.pt)
   ↓
批量预测固化 (imgsz=960) → predictions/ ⭐
   ↓
多模态 API OCR → data/ocr_results/
   ↓
馆藏目录清洗 → data/catalog/titles.json
   ↓
模糊匹配 + 去重计数 → output/inventory_result.json ⭐
   ↓
书名标注数据集 → data/book_labels/ ⭐
   ↓
FastAPI 后端 → 前端 PyQt5 单脊 OCR + 评估对比
```

## 馆藏匹配原理

匹配引擎采用 **jieba 关键词分桶 + 暴力遍历** 策略：

1. **建索引**：对 186K 馆藏书名做 jieba 分词，提取 48K+ 关键词，建立 `{关键词 → [书名字符串列表]}` 倒排索引
2. **分桶匹配**：对 OCR 名做 jieba 分词，进入对应关键词桶，桶内全量 `fuzz.ratio` 暴力遍历
3. **共现加分**：候选书名出现在多个关键词桶中，每个额外桶 +8% 分数
4. **位置加权**：书名开头位置的匹配额外加分（前 40% 字符 +0.10 线性衰减）
5. **括号降权**：全角→半角标点归一化，`core_name` 去括号后比全名权重更高
6. **策略分布**：kw_exact (62%) / kw_scan (26%) / global_ratio (<1%)
