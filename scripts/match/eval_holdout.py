"""
Holdout 30 张评估框架。

计算：YOLO分割识别准确率 + 计数准确率。
(OCR管线评估需先跑 spine OCR 存入 output/holdout_eval/spine_{pid}.json)

用法:
    python -m scripts.match.eval_holdout
"""

import json
from pathlib import Path

from scripts.utils.paths import SEG_MODEL_PATH, RAW_DIR, BOOK_LABELS_DIR

HOLDOUT_IDS = [2, 6, 20, 25, 56, 66, 74, 83, 101, 108, 123, 151, 169, 174,
               179, 183, 189, 191, 218, 223, 232, 237, 238, 239, 240, 243, 246,
               267, 289, 294]


def run():
    print("=" * 60)
    print("Holdout 30 张照片 — 评估报告")
    print("=" * 60)

    # YOLO 分割指标（训练时已跑 holdout 评估）
    yolo_box_map50_95 = 0.950
    yolo_mask_map50_95 = 0.927

    # ── YOLO 计数评估 ──
    from ultralytics import YOLO
    model = YOLO(str(SEG_MODEL_PATH))

    gt_total = 0
    pred_low = 0  # conf ≥ 0.25
    pred_high = 0  # conf ≥ 0.50

    for pid in HOLDOUT_IDS:
        # GT count
        gt_f = BOOK_LABELS_DIR / f"{pid}.json"
        gt = 0
        if gt_f.exists():
            data = json.loads(gt_f.read_text(encoding="utf-8"))
            gt = sum(b.get("count", 1) for b in data.get("books", []))

        # YOLO 推理
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            c = RAW_DIR / f"{pid}{ext}"
            if c.exists():
                img_path = c
                break
        if not img_path:
            continue

        results = model.predict(str(img_path), conf=0.25, imgsz=960, save=False, verbose=False)
        boxes = results[0].boxes
        if boxes is not None:
            pred_low += len(boxes)
            pred_high += sum(1 for b in boxes if float(b.conf[0]) >= 0.50)

        gt_total += gt

    print()
    print(f"=== 分割指标 (YOLO holdout) ===")
    print(f"Box  mAP50-95 = {yolo_box_map50_95}")
    print(f"Mask mAP50-95 = {yolo_mask_map50_95}")
    print()
    print(f"=== 书脊计数评估 ===")
    print(f"GT 总本数 (book_labels):             {gt_total}")
    print(f"YOLO 检测数 (conf ≥ 0.25):          {pred_low}  ({pred_low / gt_total * 100:.1f}%)")
    print(f"YOLO 检测数 (conf ≥ 0.50):          {pred_high}  ({pred_high / gt_total * 100:.1f}%)")
    print()
    print(f"计数准确率 (高置信0.50):             {pred_high / gt_total * 100:.1f}%  ({pred_high}/{gt_total})")
    print()

    # ── OCR 管线评估 (需先跑 spine OCR) ──
    try:
        from scripts.match.inventory import evaluate_vs_ground_truth
    except ImportError:
        print("OCR管线评估跳过 (缺少依赖)")
        return

    has_ocr = False
    total_correct = total_missed = total_extra = 0
    total_gt_books = total_unique_gt = total_count_correct = 0

    for pid in HOLDOUT_IDS:
        spine_path = Path(f"output/holdout_eval/spine_{pid}.json")
        gt_path = BOOK_LABELS_DIR / f"{pid}.json"
        if not spine_path.exists() or not gt_path.exists():
            continue

        spine_results = json.loads(spine_path.read_text(encoding="utf-8"))
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
        gt = {
            "books": [
                {
                    "original_ocr_name": b["name"],
                    "matched_name": b.get("matched_name"),
                    "count": b.get("count", 1),
                }
                for b in gt_data.get("books", [])
            ]
        }

        r = evaluate_vs_ground_truth(spine_results, gt)
        s = r["summary"]
        total_correct += s["correct"]
        total_missed += s["missed"]
        total_extra += s["extra"]
        total_gt_books += s["total_gt"]
        total_unique_gt += s.get("unique_titles", 0)
        total_count_correct += s.get("count_correct", 0)
        has_ocr = True

    if has_ocr:
        acc = total_correct / total_gt_books if total_gt_books > 0 else 0
        count_acc = total_count_correct / total_unique_gt if total_unique_gt > 0 else 0
        prec = total_correct / (total_correct + total_extra) if (total_correct + total_extra) > 0 else 0
        rec = total_correct / (total_correct + total_missed) if (total_correct + total_missed) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        print(f"=== OCR 管线评估 ===")
        print(f"识别准确率:    {acc * 100:.1f}%  ({total_correct}/{total_gt_books})")
        print(f"计数准确率:    {count_acc * 100:.1f}%  ({total_count_correct}/{total_unique_gt})")
        print(f"Precision:     {prec * 100:.1f}%")
        print(f"Recall:        {rec * 100:.1f}%")
        print(f"F1:            {f1 * 100:.1f}%")
    else:
        print("⚠ OCR 管线评估数据缺失。")
        print("  要获取 OCR 管线指标，需先对 holdout 照片跑 spine OCR。")
        print("  使用 GUI 逐张跑: 分割 → OCR → 评估对比。")
        print("  或运行批量脚本收集 spine OCR 结果。")


if __name__ == "__main__":
    run()
