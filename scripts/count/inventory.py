"""
馆藏目录匹配 + 去重计数模块。

流程:
    1. 加载 OCR 识别结果 (data/ocr_results/)
    2. 加载馆藏目录 (data/catalog/titles.json)
    3. 模糊匹配书名 → 纠正 OCR 错误
    4. 统计每本书的数量（去重计数）
    5. 输出盘点结果 (data/inventory_result.json)

用法:
    python -m scripts.count.inventory
    python -m scripts.count.inventory --threshold 0.6
"""

import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.paths import OCR_RESULTS_DIR, CATALOG_DIR

CATALOG_FILE = CATALOG_DIR / "titles.json"
RESULT_FILE = OCR_RESULTS_DIR.parent / "inventory_result.json"


def load_ocr_results() -> list[dict]:
    results = []
    if not OCR_RESULTS_DIR.exists():
        return results
    for f in sorted(OCR_RESULTS_DIR.glob("*.json")):
        try:
            results.append(json.loads(f.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, Exception):
            continue
    return results


def load_catalog() -> list[str]:
    if not CATALOG_FILE.exists():
        print(f"Error: {CATALOG_FILE} not found.")
        print("  Run: python scripts/count/extract_titles.py to generate it from xlsx")
        sys.exit(1)
    return json.loads(CATALOG_FILE.read_text(encoding="utf-8"))


def fuzzy_match(ocr_name: str, catalog: list[str], threshold: float = 0.6) -> str | None:
    """模糊匹配 OCR 书名到馆藏目录。"""
    if not ocr_name or ocr_name.startswith("未识别"):
        return None

    ocr_clean = ocr_name.strip().lower()

    best_score = 0.0
    best_match = None
    for title in catalog:
        title_clean = title.strip().lower()
        if ocr_clean == title_clean:
            return title
        if ocr_clean in title_clean or title_clean in ocr_clean:
            score = max(len(ocr_clean), len(title_clean)) / min(len(ocr_clean), len(title_clean))
            if score > best_score:
                best_score = min(score, 1.0)
                best_match = title
                continue

    if best_match and best_score >= threshold:
        return best_match

    best_score = 0.0
    for title in catalog:
        title_clean = title.strip().lower()
        score = SequenceMatcher(None, ocr_clean, title_clean).ratio()
        if score > best_score:
            best_score = score
            best_match = title

    if best_match and best_score >= threshold:
        return best_match

    return None


def count_books(results: list[dict], catalog: list[str], threshold: float = 0.6) -> dict:
    """统计每本书的数量，进行馆藏匹配和去重。"""
    book_counts: dict[str, int] = {}
    match_log: list[dict] = []

    for photo_result in results:
        for book in photo_result.get("books", []):
            ocr_name = book.get("book_name", "")
            count = book.get("count", 1)
            matched = fuzzy_match(ocr_name, catalog, threshold)

            entry = {
                "photo_id": photo_result.get("photo_id"),
                "ocr_name": ocr_name,
                "matched_name": matched,
                "count": count,
            }
            match_log.append(entry)

            display_name = matched if matched else ocr_name
            book_counts[display_name] = book_counts.get(display_name, 0) + count

    return {"book_counts": book_counts, "match_log": match_log}


def main():
    parser = argparse.ArgumentParser(description="馆藏目录匹配 + 去重计数")
    parser.add_argument("--threshold", type=float, default=0.6, help="模糊匹配阈值 (0-1)")
    args = parser.parse_args()

    results = load_ocr_results()
    if not results:
        print("No OCR results found. Run: python -m scripts.api_ocr.ocr_pipeline")
        return

    catalog = load_catalog()
    print(f"Catalog: {len(catalog)} unique titles")
    print(f"OCR results: {len(results)} photos")

    inventory = count_books(results, catalog, args.threshold)

    total_books = sum(inventory["book_counts"].values())
    matched = sum(1 for m in inventory["match_log"] if m["matched_name"])
    print(f"Matched: {matched}/{len(inventory['match_log'])} books to catalog")
    print(f"Total book copies: {total_books}")
    print(f"Unique titles: {len(inventory['book_counts'])}")
    print()

    top20 = sorted(inventory["book_counts"].items(), key=lambda x: -x[1])[:20]
    print("Top 20 books:")
    for name, count in top20:
        print(f"  {name}: {count}册")

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {RESULT_FILE}")


if __name__ == "__main__":
    main()