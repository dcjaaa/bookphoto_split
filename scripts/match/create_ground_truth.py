"""
从 OCR 识别结果生成测评基准（ground truth）。

对 data/ocr_results/ 里的每个书名跑馆藏模糊匹配：
  - 匹配分高 → status=confirmed（自动确认）
  - 有匹配但分低/歧义 → status=needs_review（需人工核实）
  - 馆藏无匹配 → status=unmatched（需人工查证）

输出: data/ground_truth/{n}.json

人工审查方式:
  打开 JSON 文件，找到 status="needs_review" 或 "unmatched" 的条目，
  修正 matched_name（改对或改 null），改 status 为 "confirmed" 或 "wrong_ocr"。

用法:
    python -m scripts.match.create_ground_truth
    python -m scripts.match.create_ground_truth --threshold 0.7
"""

import argparse
import json
import sys
from collections import Counter

from scripts.match.inventory import fuzzy_match, load_catalog, build_catalog_index
from scripts.utils.paths import OCR_RESULTS_DIR, GROUND_TRUTH_DIR


def create_ground_truth(threshold: float = 0.7) -> None:
    if not OCR_RESULTS_DIR.exists():
        print(f"Error: {OCR_RESULTS_DIR} not found.")
        print("  Run: python -m scripts.ocr.qwen_pipeline to generate OCR results first")
        return

    catalog = load_catalog()
    index = build_catalog_index(catalog)
    print(f"Catalog: {len(catalog)} titles")
    print(f"Threshold: {threshold}\n")

    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)

    ocr_files = sorted(OCR_RESULTS_DIR.glob("*.json"), key=lambda f: int(f.stem))
    total_confirmed = 0
    total_review = 0
    total_unmatched = 0
    total_books = 0

    for f in ocr_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        photo_id = data.get("photo_id", int(f.stem))
        image = data.get("image", f"{f.stem}.jpg")
        books = data.get("books", [])

        gt_books = []
        for book in books:
            ocr_name = book.get("book_name", "")
            count = book.get("count", 1)

            matched, score, strategy, needs_review = fuzzy_match(
                ocr_name, index, threshold, catalog=catalog
            )

            if matched and not needs_review:
                status = "confirmed"
            elif matched and needs_review:
                status = "needs_review"
            else:
                status = "unmatched"

            gt_books.append({
                "original_ocr_name": ocr_name,
                "matched_name": matched,
                "score": round(score, 4),
                "strategy": strategy,
                "needs_review": needs_review,
                "count": count,
                "status": status,
            })

            total_books += 1
            if status == "confirmed":
                total_confirmed += 1
            elif status == "needs_review":
                total_review += 1
            else:
                total_unmatched += 1

        gt_data = {
            "photo_id": photo_id,
            "image": image,
            "books": gt_books,
            "summary": {
                "total_books": len(gt_books),
                "confirmed": sum(1 for b in gt_books if b["status"] == "confirmed"),
                "needs_review": sum(1 for b in gt_books if b["status"] == "needs_review"),
                "unmatched": sum(1 for b in gt_books if b["status"] == "unmatched"),
            },
        }

        out_path = GROUND_TRUTH_DIR / f"{photo_id}.json"
        out_path.write_text(json.dumps(gt_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"=== 测评基准生成完成 ===")
    print(f"总文件: {len(ocr_files)}")
    print(f"总书名条目: {total_books}")
    print(f"  confirmed (自动确认):    {total_confirmed} ({total_confirmed/total_books*100:.1f}%)")
    print(f"  needs_review (需人工核实): {total_review} ({total_review/total_books*100:.1f}%)")
    print(f"  unmatched (未匹配):       {total_unmatched} ({total_unmatched/total_books*100:.1f}%)")
    print(f"\n人工审查工作量: {total_review + total_unmatched} 条需核实")
    print(f"输出: {GROUND_TRUTH_DIR}/")

    strategy_dist = Counter()
    for f in ocr_files:
        data = json.loads((GROUND_TRUTH_DIR / f.name).read_text(encoding="utf-8"))
        for b in data["books"]:
            if b["matched_name"]:
                strategy_dist[b["strategy"]] += 1
    if strategy_dist:
        print(f"\n匹配策略分布:")
        for s, n in strategy_dist.most_common():
            print(f"  {s}: {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从OCR结果生成测评基准(ground truth)")
    parser.add_argument("--threshold", type=float, default=0.7, help="模糊匹配阈值 (0-1)")
    args = parser.parse_args()
    create_ground_truth(args.threshold)
