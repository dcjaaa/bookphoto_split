"""
批量书脊分割预测：用训练好的 YOLO 模型对 raw 图片做推理，保存可视化/坐标/labelme/裁剪。

产物 (predictions/):
    vis/{id}.jpg        掩码+框可视化图
    labels/{id}.txt     YOLO 归一化坐标
    labelme/{id}.json   labelme 多边形格式 (与 data/annotations/ 一致)
    crops/{id}/spine_{n}.png  裁剪书脊 (透明 PNG)

用法:
    python -m scripts.infer.predict                       # 全部
    python -m scripts.infer.predict --start 1 --end 10
    python -m scripts.infer.predict --photo_id 5
    python -m scripts.infer.predict --conf 0.3            # 调阈值
    python -m scripts.infer.predict --force               # 强制重跑
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.utils.paths import RAW_DIR, SEG_MODEL_PATH, PRED_VIS_DIR, PRED_LABELS_DIR, PRED_LABELME_DIR, PRED_CROPS_DIR
from scripts.utils.crop import crop_with_mask

IMG_EXTS = (".jpg", ".jpeg", ".png")
LABELME_VERSION = "6.3.1"


def get_raw_image(photo_id: int) -> Path | None:
    for ext in IMG_EXTS:
        p = RAW_DIR / f"{photo_id}{ext}"
        if p.exists():
            return p
    return None


def mask_to_polygon(mask: np.ndarray) -> list[list[float]] | None:
    """从 mask 生成多边形 (xy 格式)，与 crop_spines 对齐。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.004 * cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(polygon) < 3:
        return None
    return polygon.tolist()


def draw_annotations(img: np.ndarray, spines: list[dict]) -> np.ndarray:
    """在 CPU 上绘制掩码+框+标签，避免 GPU OOM。"""
    annotated = img.copy()
    colors = []
    rng = np.random.default_rng(42)
    for _ in spines:
        colors.append((int(rng.integers(50, 255)), int(rng.integers(50, 255)), int(rng.integers(50, 255))))

    overlay = annotated.copy()
    for s, color in zip(spines, colors):
        if s["mask"] is not None:
            colored = np.zeros_like(annotated)
            colored[:] = color
            overlay = np.where(s["mask"][:, :, None] > 0, colored, overlay)
    annotated = cv2.addWeighted(annotated, 0.6, overlay, 0.4, 0)

    for s, color in zip(spines, colors):
        x1, y1, x2, y2 = s["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{s['label']} {s['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return annotated


def predict_one(model, img_path: Path, conf: float, imgsz: int) -> dict:
    """对单张图推理，返回预测结果。"""
    img = cv2.imread(str(img_path))
    if img is None:
        return {"error": "cannot read image"}
    h, w = img.shape[:2]

    results = model.predict(source=str(img_path), conf=conf, imgsz=imgsz, save=False, retina_masks=False)
    r = results[0]

    spines = []
    if r.boxes is None:
        return {"spines": spines, "height": h, "width": w, "image": img}

    masks_xy = r.masks.xy if r.masks is not None else []
    for i, box in enumerate(r.boxes):
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cls = int(box.cls[0])
        conf_val = round(float(box.conf[0]), 4)
        label = model.names[cls]

        polygon = None
        mask = None
        if i < len(masks_xy):
            poly = np.array(masks_xy[i], dtype=np.float32)
            if len(poly) >= 3:
                polygon = poly.tolist()
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [poly.astype(np.int32)], 255)

        if polygon is None:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        spines.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf_val,
            "label": label,
            "polygon": polygon,
            "mask": mask,
        })

    return {"spines": spines, "height": h, "width": w, "image": img}


def save_yolo_labels(spines: list[dict], w: int, h: int, out_path: Path):
    """保存 YOLO 归一化坐标。"""
    lines = []
    for s in spines:
        poly = s["polygon"]
        if not poly:
            x1, y1, x2, y2 = s["bbox"]
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        coords = []
        for x, y in poly:
            coords.append(f"{x / w:.6f}")
            coords.append(f"{y / h:.6f}")
        lines.append(f"0 " + " ".join(coords))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_labelme_json(spines: list[dict], img_path: Path, w: int, h: int, out_path: Path):
    """保存 labelme JSON (与 sam.py 格式一致)。"""
    shapes = []
    for s in spines:
        poly = s["polygon"]
        if not poly:
            x1, y1, x2, y2 = s["bbox"]
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        shapes.append({
            "label": "book",
            "points": poly,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {},
        })
    data = {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imageHeight": h,
        "imageWidth": w,
        "imagePath": img_path.name,
        "imageData": None,
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_crops(img: np.ndarray, spines: list[dict], out_dir: Path):
    """裁剪书脊到 crops/{id}/spine_{n}.png (透明 PNG)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(spines):
        if s["mask"] is None:
            continue
        crop = crop_with_mask(img, s["mask"], tuple(s["bbox"]), alpha=True)
        if crop is not None:
            cv2.imwrite(str(out_dir / f"spine_{i:03d}.png"), crop)


def run_batch(start: int = 1, end: int | None = None, photo_id: int | None = None,
              conf: float = 0.25, imgsz: int = 960, force: bool = False):
    if not SEG_MODEL_PATH.exists():
        print(f"Error: 模型不存在 {SEG_MODEL_PATH}")
        return

    for d in (PRED_VIS_DIR, PRED_LABELS_DIR, PRED_LABELME_DIR, PRED_CROPS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    images = sorted(
        [f for f in RAW_DIR.iterdir() if f.suffix.lower() in IMG_EXTS and f.stem.isdigit()],
        key=lambda f: int(f.stem),
    )
    if not images:
        print(f"No images in {RAW_DIR}/")
        return

    if photo_id is not None:
        images = [f for f in images if int(f.stem) == photo_id]
        if not images:
            print(f"Photo {photo_id} not found")
            return
    else:
        if end is None:
            end = len(images)
        end = min(end, len(images))
        start = max(1, start)
        images = images[start - 1:end]

    print(f"Model: {SEG_MODEL_PATH.name}")
    print(f"imgsz={imgsz}, conf={conf}")
    print(f"Images: {len(images)} ({images[0].stem}~{images[-1].stem})")
    print(f"Output: {PRED_CROPS_DIR.parent}/\n")

    from ultralytics import YOLO
    model = YOLO(str(SEG_MODEL_PATH))

    total_spines = 0
    skipped = 0
    failed = 0

    for i, img_path in enumerate(images):
        pid = int(img_path.stem)
        vis_path = PRED_VIS_DIR / f"{pid}.jpg"

        if vis_path.exists() and not force:
            skipped += 1
            print(f"[{i+1}/{len(images)}] {pid} ... SKIP")
            continue

        print(f"[{i+1}/{len(images)}] {pid} ...", end=" ", flush=True)
        result = predict_one(model, img_path, conf, imgsz)

        if "error" in result:
            print(f"FAILED: {result['error']}")
            failed += 1
            continue

        spines = result["spines"]
        h, w = result["height"], result["width"]
        img = result["image"]

        annotated = draw_annotations(img, spines)
        cv2.imwrite(str(vis_path), annotated)
        save_yolo_labels(spines, w, h, PRED_LABELS_DIR / f"{pid}.txt")
        save_labelme_json(spines, img_path, w, h, PRED_LABELME_DIR / f"{pid}.json")

        save_crops(img, spines, PRED_CROPS_DIR / str(pid))

        n = len(spines)
        total_spines += n
        print(f"OK ({n} spines)")

    print(f"\nDone: {len(images) - skipped - failed} predicted, {skipped} skipped, {failed} failed")
    print(f"Total spines: {total_spines}")
    print(f"Output: {PRED_CROPS_DIR.parent}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量书脊分割预测")
    parser.add_argument("--photo_id", type=int, help="单张图片 ID")
    parser.add_argument("--start", type=int, default=1, help="起始 ID")
    parser.add_argument("--end", type=int, default=None, help="结束 ID")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=960, help="推理尺寸 (与训练一致)")
    parser.add_argument("--force", action="store_true", help="强制重跑 (忽略已有结果)")
    args = parser.parse_args()
    run_batch(args.start, args.end, args.photo_id, args.conf, args.imgsz, args.force)
