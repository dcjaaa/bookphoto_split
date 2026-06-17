# BookPhoto Split — 书脊自动分割标注工具

基于 **SAM3** 的图书书脊自动检测、标注与裁剪工具链。

---

## 项目结构

```
bookphoto_split/
├── data/                          # 数据目录（本地，不上传）
│   ├── raw/                       # 原始书架照片（1.jpg ~ N.jpg）
│   ├── annotations/               # 标注输出（Labelme JSON 格式）
│   │   ├── 1.json                 # 第1张照片的标注
│   │   └── 2.json                 # ...
│   └── split/                     # 裁剪后的单独书脊图片
│       ├── 1/                     # 第1张照片中每本书的裁剪
│       │   ├── spine_000.png      # PNG 带透明通道，紧贴轮廓
│       │   └── spine_001.png
│       └── 2/
│           └── ...
├── scripts/                       # 工具脚本
│   ├── labelme_book.py            # 启动 Labelme 手动标注（含 SAM 辅助）
│   ├── auto_sam_annotate_v2.py    # SAM3 全自动批量标注 ⭐
│   ├── auto_sam_annotate.py       # V1 版（备查）
│   ├── auto_crop_annotations.py   # 标注完成后按轮廓裁剪书脊
│   ├── labelme_to_yolo_v2.py      # Labelme JSON → YOLO 分割训练集（原图+裁剪两路）
│   ├── labelme_json_to_yolo.py    # 旧版单路转换（备查）
│   ├── crop_spines.py             # 从 YOLO 预测结果裁剪书脊
│   └── crop_from_labelme.py       # 从 Labelme JSON 单张裁剪书脊
├── output/                        # 训练集输出
│   └── dataset/                   # YOLO 格式训练数据
├── models/                        # 模型权重（本地，不上传）
├── runs/                          # YOLO 训练日志/预测结果（自动生成）
├── requirements.txt               # Python 依赖
└── .gitignore                     # Git 忽略规则
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载 SAM3 模型

```bash
python -m osam pull sam3
```

### 3. 准备照片

将拍摄的书架照片放入 `data/raw/`，按数字命名：`1.jpg, 2.jpg, ...`

---

## 使用方式

### 方式 A：全自动批量标注（推荐先跑）

```bash
# 先试 5 张
python scripts/auto_sam_annotate_v2.py --start 1 --end 5

# 效果满意后跑全部
python scripts/auto_sam_annotate_v2.py
```

- 模型：SAM3，文本提示 `"book spine"`
- 输出：`data/annotations/{n}.json`（Labelme 多边形格式）
- 速度：约 15 秒/张（CPU），5 秒/张（GPU）

### 方式 B：Labelme 手动标注 + SAM 辅助

```bash
python scripts/labelme_book.py
```

在 Labelme 界面中：

| 操作 | 快捷键 |
|------|--------|
| AI Tools → SAM 启用 | 菜单选择 |
| 点一下书脊生成轮廓 | 鼠标左键 |
| 标签选择 `book` | 侧边栏 |
| 保存 | `Ctrl+S` |
| 下一张 | `D` |

### 裁剪书脊

标注完成后，按轮廓裁剪每本书：

```bash
python scripts/auto_crop_annotations.py
```

输出：`data/split/{照片编号}/spine_xxx.png`（PNG 带透明通道，多边形外透明）

### 生成 YOLO 训练集

```bash
python scripts/labelme_to_yolo_v2.py ./data/raw ./data/annotations ./output/dataset
```

生成两路训练数据：
- 原图 + 多边形标注
- 自动裁剪的单本书 + 对应标注

### 训练 YOLO 分割模型

```bash
yolo segment train model=yolo11s-seg.pt data=output/dataset/data.yaml epochs=100 batch=4 workers=0
```

---

## 脚本功能速查

| 脚本 | 功能 |
|------|------|
| `labelme_book.py` | 启动 GUI 标注界面，内置 SAM AI Tools |
| `auto_sam_annotate_v2.py` | SAM3 全自动批量标注（文本→检测→多边形） |
| `auto_crop_annotations.py` | 按多边形轮廓裁剪每本书（PNG 透明） |
| `labelme_to_yolo_v2.py` | 标注转 YOLO seg 训练集（原图+裁剪两路） |
| `crop_spines.py` | 从 YOLO 预测结果裁剪书脊 |
| `crop_from_labelme.py` | 从单张 Labelme JSON 裁剪书脊 |

---

## 工作流总览

```
拍摄照片 → data/raw/
   ↓
全自动标注（或 Labelme+SAM 手动） → data/annotations/
   ↓
按轮廓裁剪 → data/split/
   ↓
生成 YOLO 训练集 → output/dataset/
   ↓
YOLO 训练 → runs/
   ↓
预测 + 裁剪 → 每个书脊独立图片
```

---

## 依赖

- **Python 3.10+**
- **labelme** — 标注平台 + SAM 自动化管线
- **osam** — SAM/SAM2/SAM3 模型 ONNX 推理
- **ultralytics** — YOLO 训练与预测
- **opencv-python** — 图像处理
- **numpy** — 数值计算

详见 `requirements.txt`。
