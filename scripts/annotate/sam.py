"""
SAM3 全自动批量标注（与 Labelme GUI 一致的参数和算法）

使用 Labelme 内建 _automation 管线，参数与 GUI 完全对齐：
    - Prompt: score_threshold=0.01, iou_threshold=1.0, max_annotations=1000
    - 多边形: skimage.measure.find_contours + approximate_polygon
    - 去重: mask-level greedy suppression（与 GUI 一致）

输出: data/annotations/{n}.json (Labelme 多边形格式)

用法:
    python -m scripts.annotate.sam
    python -m scripts.annotate.sam --start 1 --end 5
    python -m scripts.annotate.sam --start 1 --end 5 --score 0.01
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import skimage.measure

from scripts.utils.paths import RAW_DIR, ANNOTATIONS_DIR

from labelme._automation._osam_session import OsamSession
from labelme._automation._text_detection import get_bboxes_from_texts, nms_bboxes
from labelme._automation._suppression import suppress_detections_greedy
from labelme._automation._shape_builders import Detection

# ── 与 Labelme GUI 对齐的参数 ──────────────────────────
MODEL_NAME = "sam3:latest"
TEXT_PROMPT = "book spine"
# GUI 内部参数: Prompt(iou_threshold=1.0, score_threshold=0.01, max_annotations=1000)
# 这里 score_threshold 和 iou_threshold 是 NMS 阶段用的，后续还有 greedy suppression
SCORE_THRESH = 0.01
IOU_THRESH = 1.0
MAX_ANNOTATIONS = 1000

# 多边形简化参数（与 labelme _geometry.py 对齐）
POLYGON_APPROX_TOLERANCE = 0.004

# 书脊形状过滤（后处理阶段，NMS 之后）
MIN_ASPECT_RATIO = 1.2
MIN_HEIGHT_PX = 50
MIN_AREA_RATIO = 0.001


def compute_polygon_from_mask(mask: np.ndarray) -> list[list[float]] | None:
    """从 mask 生成多边形，与 Labelme GUI 的 compute_polygon_from_mask 对齐。

    使用 skimage.measure.find_contours + approximate_polygon，
    而非 OpenCV 的 findContours + approxPolyDP。
    """
    contours = skimage.measure.find_contours(np.pad(mask, pad_width=1))
    if len(contours) == 0:
        return None

    # 取最长的轮廓
    def contour_length(c):
        diff = np.diff(c, axis=0)
        return np.linalg.norm(diff, axis=1).sum()

    contour = max(contours, key=contour_length)
    tolerance = np.ptp(contour, axis=0).max() * POLYGON_APPROX_TOLERANCE
    polygon = skimage.measure.approximate_polygon(coords=contour, tolerance=tolerance)
    polygon = np.clip(polygon, (0, 0), (mask.shape[0] - 1, mask.shape[1] - 1))
    polygon = polygon[:-1]  # 去掉与首点重复的末点

    # yx -> xy
    points = polygon[:, ::-1].tolist()
    if len(points) < 3:
        return None
    return points


def is_spine_shape(bb, img_h, img_w):
    """过滤非书脊形状。"""
    x1, y1, x2, y2 = bb
    w_box, h_box = x2 - x1, y2 - y1
    if w_box <= 0 or h_box <= 0:
        return False
    if h_box < MIN_HEIGHT_PX:
        return False
    if max(h_box, w_box) / max(min(h_box, w_box), 1) < MIN_ASPECT_RATIO:
        return False
    if (w_box * h_box) / max(img_w * img_h, 1) < MIN_AREA_RATIO:
        return False
    return True


def annotate_one(session, img_path: Path) -> dict | None:
    """对单张图片做 SAM 标注，参数与 Labelme GUI 对齐。"""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[skip] cannot read {img_path}")
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    # ── 与 GUI 一致：使用 OsamSession.run() 的 prompt 参数 ──
    # GUI 内部: Prompt(texts=["book spine"], iou_threshold=1.0, score_threshold=0.01, max_annotations=1000)
    boxes, scores, labels, masks = get_bboxes_from_texts(
        session=session,
        image=img_rgb,
        image_id=img_path.name,
        texts=[TEXT_PROMPT],
    )

    # NMS 去重（GUI 参数: iou_threshold=1.0 即不做 NMS，只做 score 过滤）
    if len(boxes) > 0:
        boxes, scores, labels, indices = nms_bboxes(
            boxes=boxes,
            scores=scores,
            labels=labels,
            iou_threshold=IOU_THRESH,
            score_threshold=SCORE_THRESH,
            max_num_detections=MAX_ANNOTATIONS,
        )
        filtered_masks = [masks[idx] for idx in indices] if masks is not None and len(indices) > 0 else None
    else:
        filtered_masks = None

    # 构建 Detection 列表
    detections = []
    for j in range(len(boxes)):
        det = Detection(
            bbox=tuple(boxes[j].tolist()),
            mask=filtered_masks[j] if filtered_masks and j < len(filtered_masks) else None,
            label=TEXT_PROMPT,
        )
        detections.append(det)

    # ── 与 GUI 一致：使用 mask-level greedy suppression ──
    # 这个函数同时考虑 IoU 和 containment（包含关系），比纯 NMS 更精确
    detections = suppress_detections_greedy(
        detections=detections,
        iou_threshold=0.5,  # greedy suppression 的 IoU 阈值
    )

    # ── 后处理：书脊形状过滤 + 多边形生成 ──
    shapes = []
    for det in detections:
        if det.bbox is None:
            continue
        if not is_spine_shape(det.bbox, orig_h, orig_w):
            continue

        if det.mask is not None:
            # 使用与 Labelme GUI 一致的 skimage 多边形生成
            polygon = compute_polygon_from_mask(det.mask)
            if polygon is None:
                # 退化为 bbox
                x1, y1, x2, y2 = det.bbox
                polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            else:
                # mask 坐标偏移到原图坐标
                if det.bbox:
                    ox, oy = int(det.bbox[0]), int(det.bbox[1])
                    polygon = [[p[0] + ox, p[1] + oy] for p in polygon]
        else:
            x1, y1, x2, y2 = det.bbox
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        shapes.append({
            "label": "book",
            "points": polygon,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {},
        })

    return {
        "version": "6.3.1",
        "flags": {},
        "shapes": shapes,
        "imageHeight": orig_h,
        "imageWidth": orig_w,
        "imagePath": img_path.name,
        "imageData": None,
    }


def run_batch(start_idx=1, end_idx=None, score=SCORE_THRESH, iou=IOU_THRESH):
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(
        [f for f in RAW_DIR.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda f: int(f.stem),
    )
    if not images:
        print(f"No images in {RAW_DIR}/")
        return

    if end_idx is None:
        end_idx = len(images)
    end_idx = min(end_idx, len(images))
    start_idx = max(1, start_idx)

    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: '{TEXT_PROMPT}' (score={score}, iou={iou}, max={MAX_ANNOTATIONS})")
    print(f"Polygon: skimage (GUI-aligned, tolerance={POLYGON_APPROX_TOLERANCE})")
    print(f"Suppression: mask-level greedy (iou=0.5)")
    print(f"Images: {start_idx}-{end_idx}/{len(images)}")
    print(f"Output: {ANNOTATIONS_DIR}/\n")

    print("[Init] Loading SAM3 via OsamSession...")
    session = OsamSession(model_name=MODEL_NAME)
    _ = session._get_or_load_model()
    print("[Init] Ready.\n")

    total_spines = 0
    skipped = 0
    for i in range(start_idx - 1, end_idx):
        filename = images[i]
        name = filename.stem
        json_path = ANNOTATIONS_DIR / f"{name}.json"

        # 断点续传：跳过已有标注
        if json_path.exists():
            existing = json.loads(json_path.read_text(encoding="utf-8"))
            if existing.get("shapes"):
                skipped += 1
                print(f"[{i + 1}/{end_idx}] {filename.name} ... SKIP (already annotated, {len(existing['shapes'])} shapes)")
                continue

        print(f"[{i + 1}/{end_idx}] {filename.name} ...", end=" ", flush=True)
        result = annotate_one(session, filename)
        if result is None:
            print("FAILED")
            continue

        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        n = len(result["shapes"])
        total_spines += n
        print(f"OK ({n} spines)")

    print(f"\nDone: {end_idx - start_idx + 1 - skipped} annotated, {skipped} skipped")
    print(f"Total spines: {total_spines}")
    print(f"Output: {ANNOTATIONS_DIR}/")
    print(f"\nNext: python -m scripts.prepare.crop_spines")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM3 auto-annotation (GUI-aligned params)")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--score", type=float, default=SCORE_THRESH, help="NMS score threshold (GUI=0.01)")
    parser.add_argument("--iou", type=float, default=IOU_THRESH, help="NMS IoU threshold (GUI=1.0)")
    args = parser.parse_args()
    run_batch(args.start, args.end, args.score, args.iou)