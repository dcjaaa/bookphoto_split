"""
馆藏目录匹配 + 去重计数模块 (rapidfuzz 加速版)。

优化:
    - rapidfuzz 替换 SequenceMatcher (快 ~100x)
    - 候选集过滤: 首字符 + 长度区间，缩小扫描范围
    - 优先用 titles_cleaned.json (清洗后的馆藏目录)
    - 主标题提取: 去版本号/副标题/丛书名，应对 OCR 书名比馆藏长的问题
    - 多策略匹配: 精确>主标题精确>前缀>包含>partial>ratio
    - match_log 含 score/strategy/needs_review 字段，便于调阈值和人工确认

流程:
    1. 加载 OCR 识别结果 (data/ocr_results/)
    2. 加载馆藏目录 (data/catalog/titles_cleaned.json)
    3. 模糊匹配书名 → 纠正 OCR 错误
    4. 统计每本书的数量（去重计数）
    5. 输出盘点结果 (output/inventory_result.json)

用法:
    python -m scripts.match.inventory
    python -m scripts.match.inventory --threshold 0.7
"""

import argparse
import json
import re
from pathlib import Path

from rapidfuzz import fuzz

from scripts.utils.paths import (
    OCR_RESULTS_DIR, CATALOG_DIR, OUTPUT_DIR,
    CATALOG_FILE, CATALOG_CLEANED_FILE, INVENTORY_RESULT_FILE,
)

CATALOG_PRIMARY = CATALOG_CLEANED_FILE
CATALOG_FILE_FALLBACK = CATALOG_FILE

MIN_NAME_LEN = 2
MAIN_MIN_LEN = 3
PREFIX_SUFFIX_TOLERANCE = 8

# 主标题分隔符: 版本号括号、丛书名冒号等
# 注意: 不含 · - （这些是书名内部连接符，切了会破坏书名）
_SEPARATORS = "（(:："
# 破折号分隔符 (单 — 和双 —— 都处理)
_DASHES = re.compile(r"—{1,2}|–{1,2}")
# 末尾版本/册次模式 (括号内)
_TAIL_VERSION = re.compile(r"[（(].*?[版册卷辑].*?[)）]\s*$")
# 末尾裸版本号 (无括号): "第X版" "第X册" "第X卷" "第X辑" "上册" "下册" "上卷" "下卷" "修订X版"
_TAIL_BARE_VERSION = re.compile(r"\s+(第.{1,3}[版册卷辑分]|上册|下册|上卷|下卷|修订.{0,3}版|增订.{0,3}版)\s*$")
# 前缀年份/版次: "2008" "2017" "2013" 等 4 位数字开头
_PREFIX_YEAR = re.compile(r"^\d{4}\s*")
# 前缀丛书名/修饰词 (在书名开头常见但非主标题的部分)
_PREFIX_SERIES = re.compile(
    r"^(冲击波英语英语?|（全新题型）|(?:新编|全新|插图本|新题型))\s*"
)
# 纯数字/年份
_PURE_DIGIT = re.compile(r"^[\d\s\-./]+$")
# 常见非主标题片段 (版本/册次/编者等)
_NOISE_PATTERN = re.compile(r"^(第.{1,3}[版册卷辑分]|上册|下册|上卷|下卷|修订|增订|主编|编著|编|著)")
# 英文停用词 (短英文词容易误匹配)
_EN_STOPWORDS = frozenset(
    "the a an and or for with from in on at to of by as is it be do if no not but "
    "book books two six one text teacher manual read new old".split()
)
# 纯 ASCII (英文)
_ASCII_ONLY = re.compile(r"^[a-zA-Z]+$")


def _is_valid_main(s: str) -> bool:
    """判断一个片段是否适合作为主标题候选。"""
    if not s or len(s) < MAIN_MIN_LEN:
        return False
    if _PURE_DIGIT.match(s):
        return False
    if _NOISE_PATTERN.match(s):
        return False
    # 英文片段: ≥4 字符且非停用词 (避免 AND/FOR/THE 误匹配)
    if _ASCII_ONLY.match(s):
        if len(s) < 4:
            return False
        if s.lower() in _EN_STOPWORDS:
            return False
    return True


def load_ocr_results() -> list[dict]:
    results = []
    if not OCR_RESULTS_DIR.exists():
        return results
    for f in sorted(OCR_RESULTS_DIR.glob("*.json"), key=lambda x: int(x.stem)):
        try:
            results.append(json.loads(f.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, Exception):
            continue
    return results


def load_catalog() -> list[str]:
    src = CATALOG_PRIMARY if CATALOG_PRIMARY.exists() else CATALOG_FILE_FALLBACK
    if not src.exists():
        print(f"Error: {src} not found.")
        print("  Run: python -m scripts.match.clean_catalog to generate cleaned titles")
        sys.exit(1)
    return json.loads(src.read_text(encoding="utf-8"))


def build_catalog_index(catalog: list[str]) -> dict:
    """按首字符建索引，加速候选集过滤。"""
    index: dict[str, list[tuple[str, int]]] = {}
    for title in catalog:
        if not title:
            continue
        key = title[0].lower()
        index.setdefault(key, []).append((title, len(title)))
    return index


def filter_candidates(ocr_name: str, index: dict) -> list[str]:
    """按首字符 + 长度区间过滤候选。首字符不匹配时放宽到第 2 字符。"""
    if not ocr_name:
        return []
    ocr_len = len(ocr_name)
    lo = max(ocr_len - 5, 1)
    hi = ocr_len + 5
    key = ocr_name[0].lower()
    candidates = [t for t, l in index.get(key, []) if lo <= l <= hi]
    if not candidates and len(ocr_name) >= 2:
        key2 = ocr_name[1].lower()
        candidates = [t for t, l in index.get(key2, []) if lo <= l <= hi]
    return candidates


def filter_candidates_for_main(main: str, index: dict) -> list[str]:
    """为主标题过滤候选：放宽长度区间（主标题比 OCR 全名短）。首字符容错同上。"""
    if not main:
        return []
    main_len = len(main)
    lo = max(main_len - 2, 1)
    hi = main_len + PREFIX_SUFFIX_TOLERANCE
    key = main[0].lower()
    candidates = [t for t, l in index.get(key, []) if lo <= l <= hi]
    if not candidates and len(main) >= 2:
        key2 = main[1].lower()
        candidates = [t for t, l in index.get(key2, []) if lo <= l <= hi]
    return candidates


def extract_main_titles(ocr_name: str) -> list[str]:
    """从 OCR 书名提取主标题候选。

    应对 OCR 书名比馆藏目录长的问题（含版本号/副标题/丛书名/编者等）。
    多策略提取:
      1. 去末尾版本/册次
      2. 破折号分隔 (— / ——)
      3. 括号/冒号分隔
      4. 空格分隔 (每个 ≥3 字的非噪声片段都是候选)
    过滤: 去纯数字/年份/版本号/册次/编者等噪声片段。
    """
    if not ocr_name:
        return []

    mains = []
    # 0. 去前缀年份/丛书名 + 去末尾版本号
    stripped = _PREFIX_YEAR.sub("", ocr_name).strip()
    stripped = _PREFIX_SERIES.sub("", stripped).strip()
    stripped = _TAIL_VERSION.sub("", stripped).strip()
    stripped = _TAIL_BARE_VERSION.sub("", stripped).strip()
    if stripped != ocr_name and _is_valid_main(stripped):
        mains.append(stripped)

    # 1. 去末尾版本/册次 (对原名也做，保留原逻辑)
    cleaned = _TAIL_VERSION.sub("", ocr_name).strip()
    cleaned = _TAIL_BARE_VERSION.sub("", cleaned).strip()
    if cleaned != ocr_name and _is_valid_main(cleaned):
        mains.append(cleaned)

    # 2. 破折号分隔 (— / —— / – / ——)
    dash_parts = _DASHES.split(ocr_name, maxsplit=1)
    if len(dash_parts) == 2:
        head = dash_parts[0].strip()
        tail = _TAIL_VERSION.sub("", dash_parts[1]).strip()
        if _is_valid_main(head):
            mains.append(head)
        if _is_valid_main(tail):
            mains.append(tail)

    # 3. 括号/冒号分隔
    bracket_parts = re.split(f"[{re.escape(_SEPARATORS)}]", ocr_name, maxsplit=1)
    if len(bracket_parts) == 2:
        head = bracket_parts[0].strip()
        tail = _TAIL_VERSION.sub("", bracket_parts[1]).strip()
        if _is_valid_main(head):
            mains.append(head)
        if _is_valid_main(tail):
            mains.append(tail)

    # 4. 空格分隔 (仅对含中文的 OCR 名，英文书名空格是标题内部的一部分)
    #    例: "刑法各罪论 上册 修订五版" → "刑法各罪论" 在馆藏中
    has_chinese = bool(re.search(r"[\u4e00-\u9fff]", ocr_name))
    if has_chinese:
        space_parts = ocr_name.split()
        if len(space_parts) > 1:
            for part in space_parts:
                part = part.strip()
                # 去末尾版本号
                part = _TAIL_VERSION.sub("", part).strip()
                if _is_valid_main(part):
                    mains.append(part)

    # 去重保序
    result = []
    seen = set()
    for m in mains:
        if m not in seen and m != ocr_name:
            seen.add(m)
            result.append(m)
    return result


def fuzzy_match(ocr_name: str, index: dict, threshold: float = 0.7, catalog: list[str] | None = None) -> tuple[str | None, float, str, bool]:
    """模糊匹配 OCR 书名到馆藏目录。

    返回 (匹配名, 分数, 策略, 是否需人工确认)。
    多策略优先级: 精确 > 主标题精确 > 前缀 > 包含 > partial > token_set > ratio > 全库兜底。
    """
    if not ocr_name or ocr_name.startswith("未识别") or len(ocr_name) < MIN_NAME_LEN:
        return None, 0.0, "skip", False

    ocr_lower = ocr_name.strip().lower()
    candidates = filter_candidates(ocr_name, index)

    # 策略1: OCR 全名精确匹配
    for title in candidates:
        if ocr_lower == title.strip().lower():
            return title, 1.0, "exact_full", False

    # 提取主标题
    main_titles = extract_main_titles(ocr_name)

    # 策略2: 主标题精确匹配馆藏
    for main in main_titles:
        main_lower = main.lower()
        main_candidates = filter_candidates_for_main(main, index)
        for title in main_candidates:
            if main_lower == title.strip().lower():
                return title, 0.98, "exact_main", False

    # 策略3: 主标题前缀匹配 (馆藏以主标题开头，后缀≤容忍度)
    # 主标题越短，容忍度越严格，避免 "Excel" 匹配到 "Excel之美" 等误匹配
    best_prefix = None
    best_prefix_suffix_len = 999
    best_prefix_multi = False
    for main in main_titles:
        main_lower = main.lower()
        main_len = len(main_lower)
        # 纯英文短词(如 Excel/SPSS)不参与 prefix_main，太容易跨书误匹配
        if _ASCII_ONLY.match(main) and main_len < 8:
            continue
        # 短主标题严格限制后缀: <4字→后缀≤2, <6字→后缀≤4, 否则≤8
        if main_len < 4:
            suffix_limit = 2
        elif main_len < 6:
            suffix_limit = 4
        else:
            suffix_limit = PREFIX_SUFFIX_TOLERANCE
        main_candidates = filter_candidates_for_main(main, index)
        prefix_hits = []
        for title in main_candidates:
            title_lower = title.strip().lower()
            if title_lower.startswith(main_lower) and len(title_lower) >= len(main_lower):
                suffix_len = len(title_lower) - len(main_lower)
                if suffix_len <= suffix_limit:
                    prefix_hits.append((title, suffix_len))
        if prefix_hits:
            prefix_hits.sort(key=lambda x: x[1])
            top_title, top_suffix = prefix_hits[0]
            if top_suffix < best_prefix_suffix_len:
                best_prefix = top_title
                best_prefix_suffix_len = top_suffix
                best_prefix_multi = len(prefix_hits) > 1

    if best_prefix:
        # 判定是否需人工确认
        needs_review = False
        if best_prefix_suffix_len >= 5:
            needs_review = True
        elif best_prefix_multi:
            # 多候选且后缀接近 (差≤2) → 歧义
            needs_review = True
        return best_prefix, 0.92, "prefix_main", needs_review

    # 策略4: 馆藏包含于主标题 (馆藏是主标题子串，馆藏≥3字)
    for main in main_titles:
        main_lower = main.lower()
        main_candidates = filter_candidates_for_main(main, index)
        best_contained = None
        best_contained_len = 0
        for title in main_candidates:
            title_lower = title.strip().lower()
            if len(title_lower) >= 3 and title_lower in main_lower:
                if len(title_lower) > best_contained_len:
                    best_contained = title
                    best_contained_len = len(title_lower)
        if best_contained:
            return best_contained, 0.88, "contained", False

    # 策略4b: 主标题包含于馆藏 (主标题是馆藏子串，主标题≥4字)
    # 救 "涉外文书写作大全" → 馆藏 "新编涉外文书写作大全" 这种馆藏有前缀的情况
    for main in main_titles:
        main_lower = main.lower()
        if len(main_lower) < 4:
            continue
        main_candidates = filter_candidates_for_main(main, index)
        best_contained_in = None
        best_contained_in_suffix = 999
        for title in main_candidates:
            title_lower = title.strip().lower()
            if len(title_lower) > len(main_lower) and main_lower in title_lower:
                suffix_len = len(title_lower) - len(main_lower)
                if suffix_len < best_contained_in_suffix:
                    best_contained_in = title
                    best_contained_in_suffix = suffix_len
        if best_contained_in:
            needs_review = best_contained_in_suffix >= 5
            return best_contained_in, 0.85, "contained_in", needs_review

    # 策略5: OCR 全名 partial_ratio
    best_score = 0.0
    best_match = None
    for title in candidates:
        title_lower = title.strip().lower()
        score = fuzz.partial_ratio(ocr_lower, title_lower) / 100.0
        if score > best_score:
            best_score = score
            best_match = title
    if best_match and best_score >= threshold:
        return best_match, best_score, "partial_full", False

    # 策略6: token_set_ratio (对空格/词序不敏感)
    for title in candidates:
        title_lower = title.strip().lower()
        score = fuzz.token_set_ratio(ocr_lower, title_lower) / 100.0
        if score > best_score:
            best_score = score
            best_match = title
    if best_match and best_score >= threshold:
        return best_match, best_score, "token_set", False

    # 策略7: 主标题 partial_ratio
    for main in main_titles:
        main_lower = main.lower()
        main_candidates = filter_candidates_for_main(main, index)
        for title in main_candidates:
            title_lower = title.strip().lower()
            score = fuzz.partial_ratio(main_lower, title_lower) / 100.0
            if score > best_score:
                best_score = score
                best_match = title
    if best_match and best_score >= threshold:
        return best_match, best_score, "partial_main", False

    # 策略8: OCR 全名 fuzz.ratio (兜底)
    for title in candidates:
        title_lower = title.strip().lower()
        score = fuzz.ratio(ocr_lower, title_lower) / 100.0
        if score > best_score:
            best_score = score
            best_match = title
    if best_match and best_score >= threshold:
        return best_match, best_score, "ratio", False

    # 策略9: 全库 fuzz.ratio 兜底 (不限首字符，救 OCR 首字错的情况)
    if catalog is not None and best_score < threshold:
        for title in catalog:
            title_lower = title.strip().lower()
            score = fuzz.ratio(ocr_lower, title_lower) / 100.0
            if score > best_score:
                best_score = score
                best_match = title
            # 也对主标题做全库 ratio
        for main in main_titles:
            main_lower = main.lower()
            if len(main_lower) < 4:
                continue
            for title in catalog:
                title_lower = title.strip().lower()
                score = fuzz.ratio(main_lower, title_lower) / 100.0
                if score > best_score:
                    best_score = score
                    best_match = title
        if best_match and best_score >= 0.70:
            needs_review = best_score < 0.85
            return best_match, best_score, "global_ratio", needs_review

    return None, best_score, "none", False


def count_books(results: list[dict], index: dict, threshold: float = 0.7, catalog: list[str] | None = None) -> dict:
    """统计每本书的数量，进行馆藏匹配和去重。"""
    book_counts: dict[str, int] = {}
    match_log: list[dict] = []

    for photo_result in results:
        for book in photo_result.get("books", []):
            ocr_name = book.get("book_name", "")
            count = book.get("count", 1)
            matched, score, strategy, needs_review = fuzzy_match(ocr_name, index, threshold, catalog=catalog)

            entry = {
                "photo_id": photo_result.get("photo_id"),
                "ocr_name": ocr_name,
                "matched_name": matched,
                "score": round(score, 4),
                "strategy": strategy,
                "needs_review": needs_review,
                "count": count,
            }
            match_log.append(entry)

            display_name = matched if matched else ocr_name
            book_counts[display_name] = book_counts.get(display_name, 0) + count

    return {"book_counts": book_counts, "match_log": match_log}


def main():
    parser = argparse.ArgumentParser(description="馆藏目录匹配 + 去重计数 (rapidfuzz + 主标题提取)")
    parser.add_argument("--threshold", type=float, default=0.7, help="模糊匹配阈值 (0-1)")
    args = parser.parse_args()

    results = load_ocr_results()
    if not results:
        print("No OCR results found. Run: python -m scripts.ocr.qwen_pipeline")
        return

    catalog = load_catalog()
    index = build_catalog_index(catalog)
    print(f"Catalog: {len(catalog)} titles (index: {len(index)} groups)")
    print(f"OCR results: {len(results)} photos")
    print(f"Threshold: {args.threshold}\n")

    import time
    t0 = time.time()
    inventory = count_books(results, index, args.threshold, catalog=catalog)
    elapsed = time.time() - t0

    total_books = sum(inventory["book_counts"].values())
    total_entries = len(inventory["match_log"])
    matched = sum(1 for m in inventory["match_log"] if m["matched_name"])
    needs_review = [m for m in inventory["match_log"] if m.get("needs_review")]
    unmatched = [m for m in inventory["match_log"] if not m["matched_name"]]

    # 策略分布
    from collections import Counter
    strategy_dist = Counter(m["strategy"] for m in inventory["match_log"] if m["matched_name"])

    print(f"耗时: {elapsed:.2f}s")
    print(f"Matched: {matched}/{total_entries} ({matched/total_entries*100:.1f}%)")
    print(f"  需人工确认: {len(needs_review)}")
    print(f"  未匹配: {len(unmatched)}")
    print(f"Total book copies: {total_books}")
    print(f"Unique titles: {len(inventory['book_counts'])}")
    print(f"\n策略分布:")
    for s, n in strategy_dist.most_common():
        print(f"  {s}: {n}")

    top20 = sorted(inventory["book_counts"].items(), key=lambda x: -x[1])[:20]
    print(f"\nTop 20 books:")
    for name, count in top20:
        print(f"  {name}: {count}册")

    if needs_review:
        print(f"\n需人工确认 (前20):")
        for m in needs_review[:20]:
            print(f"  photo {m['photo_id']}: OCR='{m['ocr_name'][:40]}' → '{m['matched_name']}' (score={m['score']}, {m['strategy']})")

    if unmatched:
        print(f"\n未匹配样例 (前20):")
        for m in unmatched[:20]:
            print(f"  photo {m['photo_id']}: '{m['ocr_name'][:40]}' (score={m['score']})")

    INVENTORY_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    INVENTORY_RESULT_FILE.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {INVENTORY_RESULT_FILE}")


if __name__ == "__main__":
    main()
