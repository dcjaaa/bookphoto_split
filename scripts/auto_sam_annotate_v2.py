"""
Labelme AI 批量标注 V2 — 使用内建 _automation 管线
参考 Labelme GUI 的 AI Text-to-Annotation 逻辑:
  OsamSession (长连接) → get_bboxes_from_texts → NMS → 贪心去重 → 掩码转多边形

输出: data/annotations/{n}.json (Labelme 多边形格式)

用法:
    python scripts/auto_sam_annotate_v2.py                    # 全部294张
    python scripts/auto_sam_annotate_v2.py --start 1 --end 5  # 先试5张
    python scripts/auto_sam_annotate_v2.py --start 1 --end 5 --score 0.3
"""
import os
import sys
import json
import cv2
import numpy as np
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from labelme._automation._osam_session import OsamSession
from labelme._automation._text_detection import get_bboxes_from_texts, nms_bboxes
from labelme._automation._suppression import suppress_detections_greedy
from labelme._automation._shape_builders import Detection

RAW_DIR = "data/raw"
ANNOTATION_DIR = "data/annotations"

# 书脊形状过滤
MIN_ASPECT_RATIO = 1.5
MIN_HEIGHT = 80
MIN_AREA_RATIO = 0.003

# 模型配置
MODEL_NAME = "sam3:latest"
TEXT_PROMPT = "book spine"
SCORE_THRESH = 0.15
IOU_THRESH = 0.5
MAX_DETECTIONS = 200


def mask_to_polygon(mask, simplify_epsilon=2.0):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = simplify_epsilon / 1000.0 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = approx.squeeze(axis=1).tolist()
    if isinstance(points, list) and len(points) > 0 and isinstance(points[0], (int, float)):
        if len(points) < 4:
            return None
        # 只有一行，可能需要重塑
        if len(points) == 4 or len(points) % 2 == 0:
            points = [[points[j], points[j+1]] for j in range(0, len(points), 2)]
            if len(points) < 4:
                return None
        else:
            return None
    return points


def is_spine_shape(bb, img_h, img_w):
    x1, y1, x2, y2 = bb
    w_box = x2 - x1
    h_box = y2 - y1
    if w_box <= 0 or h_box <= 0 or h_box < MIN_HEIGHT:
        return False
    ratio = max(h_box, w_box) / max(min(h_box, w_box), 1)
    if ratio < MIN_ASPECT_RATIO:
        return False
    if (w_box * h_box) / (img_w * img_h) < MIN_AREA_RATIO:
        return False
    return True


def run_batch(start_idx=1, end_idx=None, score=SCORE_THRESH, iou=IOU_THRESH):
    os.makedirs(ANNOTATION_DIR, exist_ok=True)

    images = sorted(
        [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    if not images:
        print(f"No images in {RAW_DIR}/")
        return

    if end_idx is None:
        end_idx = len(images)
    end_idx = min(end_idx, len(images))
    start_idx = max(1, start_idx)

    print(f"Model: {MODEL_NAME}  |  Prompt: '{TEXT_PROMPT}'")
    print(f"Score: {score}  |  IoU: {iou}  |  Images: {start_idx}-{end_idx}/{len(images)}")
    print(f"Output: {ANNOTATION_DIR}/")
    print()

    # 初始化 OsamSession (长连接)
    print("[Init] Loading SAM3 via OsamSession...")
    session = OsamSession(model_name=MODEL_NAME)
    _ = session._get_or_load_model()
    print("[Init] Ready.\n")

    total_spines = 0

    for i in range(start_idx - 1, end_idx):
        filename = images[i]
        name = os.path.splitext(filename)[0]
        img_path = os.path.join(RAW_DIR, filename)
        json_path = os.path.join(ANNOTATION_DIR, f"{name}.json")

        print(f"[{i+1}/{end_idx}] {filename} ...", end=" ")

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print("[skip] cannot read")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Step 1: 文本 → bbox + masks
        boxes, scores, labels, masks = get_bboxes_from_texts(
            session=session,
            image=img_rgb,
            image_id=filename,
            texts=[TEXT_PROMPT],
        )

        # Step 2: NMS 去重
        if len(boxes) > 0:
            boxes, scores, labels, indices = nms_bboxes(
                boxes=boxes,
                scores=scores,
                labels=labels,
                iou_threshold=iou,
                score_threshold=score,
                max_num_detections=MAX_DETECTIONS,
            )
            filtered_masks = [masks[i] for i in indices] if masks and len(indices) > 0 else None
        else:
            filtered_masks = None

        # Step 3: 构建 Detection 列表
        detections = []
        for j in range(len(boxes)):
            det = Detection(
                bbox=tuple(boxes[j].tolist()),
                mask=filtered_masks[j] if filtered_masks and j < len(filtered_masks) else None,
                label=TEXT_PROMPT,
            )
            detections.append(det)

        # Step 4: 贪心去重
        detections = suppress_detections_greedy(
            detections=detections,
            iou_threshold=iou,
        )

        # Step 5: 掩码 → 多边形 + 书脊过滤
        shapes = []
        for det in detections:
            if det.bbox is None:
                continue
            if not is_spine_shape(det.bbox, h, w):
                continue
            if det.mask is None:
                # 无掩码时用 bbox 造一个矩形polygon
                x1, y1, x2, y2 = det.bbox
                polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            else:
                polygon = mask_to_polygon(det.mask)
                if polygon is None:
                    continue
                # mask 坐标是局部的，偏移到原图坐标
                if det.bbox:
                    ox, oy = int(det.bbox[0]), int(det.bbox[1])
                    polygon = [[p[0] + ox, p[1] + oy] for p in polygon]

            shapes.append({
                "label": "book",
                "points": polygon,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            })

        result = {
            "version": "6.3.1",
            "flags": {},
            "shapes": shapes,
            "imageHeight": h,
            "imageWidth": w,
            "imagePath": filename,
            "imageData": None,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        total_spines += len(shapes)
        print(f"OK ({len(shapes)} spines)")

    print(f"\nDone: {end_idx-start_idx+1} images -> {total_spines} book spines")
    print(f"Output: {ANNOTATION_DIR}/")
    print(f"\nNext: python scripts/auto_crop_annotations.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Labelme AI auto-annotation V2")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--score", type=float, default=SCORE_THRESH, help="Score threshold")
    parser.add_argument("--iou", type=float, default=IOU_THRESH, help="NMS IoU threshold")
    args = parser.parse_args()

    run_batch(args.start, args.end, args.score, args.iou)
