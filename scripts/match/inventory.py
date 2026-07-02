"""
馆藏目录匹配 + 去重计数模块 (rapidfuzz 加速版)。

优化:
    - rapidfuzz 替换 SequenceMatcher (快 ~100x)
    - 候选集过滤: 首字符 + 长度区间，缩小扫描范围
    - 主标题提取: 去版本号/副标题/丛书名，应对 OCR 书名比馆藏长的问题
    - 多策略匹配: 精确>主标题精确>前缀>包含>partial>ratio
    - match_log 含 score/strategy/needs_review 字段，便于调阈值和人工确认

流程:
    1. 加载 OCR 识别结果 (data/ocr_results/)
    2. 加载馆藏目录 (data/catalog/titles.json)
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

import jieba
from rapidfuzz import fuzz

from scripts.utils.paths import (
    OCR_RESULTS_DIR, CATALOG_DIR, OUTPUT_DIR,
    CATALOG_FILE, INVENTORY_RESULT_FILE,
)

# 关键词索引缓存 (惰性加载)
_keyword_index_cache: dict | None = None
_keyword_title_cache: dict[str, list[str]] | None = None  # keyword → [title, ...]


def _load_keyword_index() -> dict[str, list[str]]:
    """加载 jieba 关键词索引。首次调用从 JSON 加载，后续用缓存。"""
    global _keyword_title_cache
    if _keyword_title_cache is not None:
        return _keyword_title_cache

    idx_path = CATALOG_DIR / "keyword_index.json"
    if not idx_path.exists():
        print("[warn] keyword_index.json 未找到，请先运行 build_keyword_index")
        _keyword_title_cache = {}
        return _keyword_title_cache

    import json as _json
    data = _json.loads(idx_path.read_text(encoding="utf-8"))
    kw_map = data.get("keywords", {})
    catalog = load_catalog()
    # 将 title_idx 转为 title 字符串
    result: dict[str, list[str]] = {}
    for kw, indices in kw_map.items():
        titles = []
        for idx in indices:
            if idx < len(catalog):
                titles.append(catalog[idx])
        result[kw] = titles
    _keyword_title_cache = result
    return _keyword_title_cache

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
    if not CATALOG_FILE.exists():
        print(f"Error: {CATALOG_FILE} not found.")
        print("  Run: python -m scripts.match.clean_catalog to generate cleaned titles")
        sys.exit(1)
    return json.loads(CATALOG_FILE.read_text(encoding="utf-8"))


def build_catalog_index(catalog: list[str]) -> dict:
    """按首字符 + 前缀分桶(前2有效汉字)建索引，加速候选集过滤。"""
    index: dict[str, list[tuple[str, int]]] = {}
    for title in catalog:
        if not title:
            continue
        key = title[0].lower()
        index.setdefault(key, []).append((title, len(title)))
    # 附加前缀分桶索引 (前2有效汉字)
    index["_prefix2"] = _build_prefix2_index(catalog)
    return index


def _clean_prefix(name: str, n: int = 2) -> str:
    """取前 n 个有效字符(去数字/英文/标点/空格)，用于分桶键。"""
    cleaned = re.sub(
        r'[\s\dA-Za-z\.\,\;\:\!\?\(\)\[\]\{\}（）【】《》〈〉「」『』\"\'—…·★☆●◆\-/\\&@#\$%\^&\*\+\=~`#]+',
        '', name,
    )
    if len(cleaned) >= n:
        return cleaned[:n]
    return cleaned if cleaned else name[:n]


def _build_prefix2_index(catalog: list[str]) -> dict[str, list[tuple[str, int]]]:
    """建前缀分桶索引：前2有效汉字 → [(title, length), ...]"""
    idx: dict[str, list[tuple[str, int]]] = {}
    for title in catalog:
        if not title:
            continue
        key = _clean_prefix(title)
        idx.setdefault(key, []).append((title, len(title)))
    return idx


def _keyword_candidates(ocr_name: str, index: dict) -> list[str] | None:
    """用 jieba 分词找到最小的关键词桶作为候选池。返回 None 表示无合适桶。"""
    kt = _load_keyword_index()
    if not kt:
        return None

    words = list(jieba.cut(ocr_name))
    best_bucket = None
    best_size = 99999

    for w in words:
        w = w.strip()
        if not w or len(w) <= 1:
            continue
        bucket = kt.get(w)
        if bucket and len(bucket) < best_size:
            # 跳过太通用的词(>1000) — 不是有效候选桶
            if len(bucket) <= 1000:
                best_bucket = bucket
                best_size = len(bucket)

    return best_bucket


def filter_candidates(ocr_name: str, index: dict) -> list[str]:
    """优选关键词桶 → 前缀桶 → 首字符兜底，三级候选过滤。"""
    if not ocr_name:
        return []
    ocr_len = len(ocr_name)
    lo = max(ocr_len - 5, 1)
    hi = ocr_len + 5

    # 第1级: jieba 关键词桶 (最小最精确)
    kw_cand = _keyword_candidates(ocr_name, index)
    if kw_cand:
        candidates = [t for t in kw_cand if lo <= len(t) <= hi]
        if candidates:
            return candidates

    # 第2级: 前缀分桶 (前2有效汉字)
    prefix2_idx = index.get("_prefix2", {})
    if prefix2_idx:
        key = _clean_prefix(ocr_name)
        bucket = prefix2_idx.get(key, [])
        if bucket:
            candidates = [t for t, l in bucket if lo <= l <= hi]
            if candidates:
                return candidates

    # 第3级: 首字符兜底
    key = ocr_name[0].lower()
    candidates = [t for t, l in index.get(key, []) if lo <= l <= hi]
    if not candidates and len(ocr_name) >= 2:
        key2 = ocr_name[1].lower()
        candidates = [t for t, l in index.get(key2, []) if lo <= l <= hi]
    return candidates


def filter_candidates_for_main(main: str, index: dict) -> list[str]:
    """为主标题过滤候选：三级策略(关键词桶 → 前缀桶 → 首字符兜底)。"""
    if not main:
        return []
    main_len = len(main)
    lo = max(main_len - 2, 1)
    hi = main_len + PREFIX_SUFFIX_TOLERANCE

    # 第1级: jieba 关键词桶
    kw_cand = _keyword_candidates(main, index)
    if kw_cand:
        candidates = [t for t in kw_cand if lo <= len(t) <= hi]
        if candidates:
            return candidates

    # 第2级: 前缀桶
    prefix2_idx = index.get("_prefix2", {})
    if prefix2_idx:
        key = _clean_prefix(main)
        bucket = prefix2_idx.get(key, [])
        if bucket:
            candidates = [t for t, l in bucket if lo <= l <= hi]
            if candidates:
                return candidates

    # 第3级: 首字符兜底
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

    # 策略0: jieba 关键词分桶暴力遍历
    # 统计每个候选出现在几个桶(共现)，结合 fuzz.ratio 排序
    # 括号内容降权: 核心名(去括号)比全名权重大
    kt = _load_keyword_index()
    if kt:
        import re as _re
        # 核心名: （）括号去内容，其他符号只去壳保留内容
        core_name = _re.sub(r'[（(][^)）]*[)）]', '', ocr_name)  # 去版本号/注释
        core_name = _re.sub(r'[〈《「『\"\'』」〉》]', '', core_name)  # 去书名号/引号壳
        core_name = core_name.strip()
        core_lower = core_name.lower() if core_name else ocr_lower

        candidate_scores: dict[str, tuple[float, int]] = {}  # title → (best_fuzz, hit_count)
        words = list(jieba.cut(ocr_name))
        for w in words:
            w = w.strip()
            if not w or len(w) <= 1:
                continue
            bucket = kt.get(w)
            if not bucket:
                continue
            for title in bucket:
                prev = candidate_scores.get(title)
                if prev:
                    prev_fuzz, prev_hits = prev
                else:
                    prev_fuzz, prev_hits = 0.0, 0

                # 核心名比例权重 + 直接包含检测
                core_r = fuzz.ratio(core_lower, title.lower()) / 100.0
                full_r = fuzz.ratio(ocr_lower, title.lower()) / 100.0
                ts_r = fuzz.token_set_ratio(ocr_lower, title.lower()) / 100.0
                # 馆藏名是 OCR 名子串(≥2字, 去空格+标点归一化后比较) → 至少 0.75
                # 出现在分隔符(：——)前(+0.05)或开头位置也有小加成
                contained_bonus = 0.0
                # 标点归一化: 全角↔半角 (!↔！ ,？↔? "↔“” 等)
                _norm_tbl = str.maketrans(
                    "！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～",
                    '!"#$%&' + "'()*+,-./:;<=>?@[\\]^_`{|}~"
                )
                title_nosp = title.lower().replace(" ", "").translate(_norm_tbl)
                ocr_nosp = ocr_lower.replace(" ", "").translate(_norm_tbl)
                core_nosp = core_lower.replace(" ", "").translate(_norm_tbl)
                if len(title_nosp) >= 2:
                    pos = ocr_nosp.find(title_nosp)
                    if pos < 0:
                        pos = core_nosp.find(title_nosp)
                    if pos >= 0:
                        contained_bonus = 0.75
                        # 分隔符前(正标题) → +0.05
                        sep_pos = min(
                            ocr_nosp.find("：") if ocr_nosp.find("：") >= 0 else 99999,
                            ocr_nosp.find("——") if ocr_nosp.find("——") >= 0 else 99999,
                            ocr_nosp.find("—") if ocr_nosp.find("—") >= 0 else 99999,
                        )
                        if sep_pos < 99999 and pos < sep_pos:
                            contained_bonus += 0.05
                        # 开头加权: pos=0 → +0.10, 线性衰减到末尾 0
                        if len(ocr_nosp) > 0:
                            contained_bonus += 0.10 * max(0, 1.0 - pos / max(len(ocr_nosp) * 0.4, 1))
                fuzz_score = max(core_r, full_r * 0.85, ts_r, contained_bonus)

                if fuzz_score > prev_fuzz:
                    prev_fuzz = fuzz_score

                candidate_scores[title] = (prev_fuzz, prev_hits + 1)

                # 精确命中直接返回
                if core_r >= 0.99:
                    return title, 1.0, "kw_exact", False

        # 找最佳：fuzz 分 + 共现加分(每多一个桶+8%)，同分时取位置靠前的
        best_kw_score = 0.0
        best_kw_match = None
        best_kw_pos = 99999  # 在 OCR 名中的位置
        for title, (fuzz_s, hit_n) in candidate_scores.items():
            combined = fuzz_s * (1.0 + 0.08 * (hit_n - 1))
            title_nosp = title.lower().replace(" ", "")
            pos = ocr_lower.replace(" ", "").find(title_nosp)
            if pos < 0:
                pos = core_lower.replace(" ", "").find(title_nosp)
            if pos < 0:
                pos = 99999
            # 同分或分更高时选位置更靠前的
            if combined > best_kw_score or (
                abs(combined - best_kw_score) < 0.001 and pos < best_kw_pos
            ):
                best_kw_score = combined
                best_kw_match = title
                best_kw_pos = pos

        if best_kw_match and best_kw_score >= 0.60:
            needs_review = fuzz_s < 0.80  # 用原始 fuzz 分判断
            return best_kw_match, best_kw_score, "kw_scan", needs_review

    # 兜底: 关键词桶无匹配时，仅当 kw 有弱信号才做全库 ratio
    if catalog is not None and best_kw_score > 0.35:
        best_score = 0.0
        best_match = None
        for title in catalog:
            score = max(
                fuzz.ratio(ocr_lower, title.lower()),
                fuzz.token_set_ratio(ocr_lower, title.lower()),
            ) / 100.0
            if score > best_score:
                best_score = score
                best_match = title
        if best_match and best_score >= 0.70:
            return best_match, best_score, "global_ratio", True

    return None, 0.0, "none", False


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


FUZZY_EVAL_THRESHOLD = 0.85  # OCR 评估: 多策略匹配的兜底阈值


def _eval_match_score(ocr_name: str, gt_name: str) -> float:
    """多策略计算单脊OCR名 vs GT原始书名的匹配分。

    - 精确相等 → 1.0
    - 包含关系，且长度比 < 0.6 (片段 vs 完整书名) → 不低于 0.92
    - 其他情况取 fuzz.ratio，并用 partial/token 小幅提升(≤1.25x)
    """
    a, b = ocr_name.lower().strip(), gt_name.lower().strip()
    if a == b:
        return 1.0

    # 标点归一化: 全角→半角
    _nt = str.maketrans(
        "！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～",
        '!"#$%&' + "'()*+,-./:;<=>?@[\\]^_`{|}~"
    )
    a, b = a.translate(_nt), b.translate(_nt)
    if a == b:
        return 1.0

    ratio = fuzz.ratio(a, b) / 100.0
    len_ratio = min(len(a), len(b)) / max(len(a), len(b))

    # 包含关系: 一方显著短 → 是另一方的片段 (且至少4字, 防单字假阳性)
    if (a in b or b in a) and len_ratio < 0.6 and min(len(a), len(b)) >= 4:
        return max(ratio, 0.92)

    # OCR⊂GT: OCR 漏字后不再是连续子串, 用 partial_ratio 补救
    # 短串≥3字且 partial_ratio ≥ 0.92 → 至少 0.85 (防"书"⊂"一大堆书"假阳性)
    if min(len(a), len(b)) >= 3:
        pr = fuzz.partial_ratio(a, b) / 100.0
        if pr >= 0.92:
            return max(ratio, 0.85)

    # 相似长度的串: ratio 为基础，partial/token 小幅提升
    cap = ratio * 1.15
    pr = fuzz.partial_ratio(a, b) / 100.0
    ts = fuzz.token_set_ratio(a, b) / 100.0
    return max(ratio, min(pr, cap), min(ts, cap))


def evaluate_vs_ground_truth(
    spine_results: list[dict],
    ground_truth: dict,
) -> dict:
    """
    两轮匹配：先比 OCR 原名，败退后比馆藏匹配名。

    第1轮: spine.book_name vs GT.original_ocr_name
    第2轮: spine.matched_name vs GT.matched_name (兜底)
    书名标注列显示 GT 的 matched_name。

    Returns:
        per_spine: [{spine_idx, ocr_name, matched_name, gt_name, gt_score, result}]
        summary: {correct, missed, extra, total_gt, total_spine, accuracy}
    """
    gt_books = ground_truth.get("books", [])

    # GT池: (original_ocr_name, matched_name) × count
    gt_pool: list[tuple[str, str | None]] = []
    for b in gt_books:
        orig = b.get("original_ocr_name", "")
        mt = b.get("matched_name")
        count = b.get("count", 1)
        for _ in range(count):
            gt_pool.append((orig, mt))

    per_spine = []
    correct = 0
    extra = 0

    for sr in spine_results:
        book_name = sr.get("book_name", "")
        matched = sr.get("matched_name")
        spine_idx = sr.get("spine_idx", 0)

        if sr.get("strategy") == "skip_conf":
            per_spine.append({
                "spine_idx": spine_idx, "ocr_name": book_name,
                "matched_name": matched, "gt_name": None, "gt_score": 0.0,
                "gt_matched": None, "match_round": "skip",
                "result": "skipped",
            })
            continue

        result = "extra"
        gt_display_name = None
        gt_catalog_name = None  # GT 对应的馆藏名
        match_round = ""
        best_score = 0.0
        best_idx = -1

        # 第1轮: spine book_name vs GT original_ocr_name
        for i, (orig, mt) in enumerate(gt_pool):
            score = _eval_match_score(book_name, orig)
            if score > best_score:
                best_score = score
                best_idx = i
                gt_display_name = orig  # 显示 book_labels 的 name
                gt_catalog_name = mt     # 对应的馆藏名

        if best_score >= FUZZY_EVAL_THRESHOLD and best_idx >= 0:
            gt_pool.pop(best_idx)
            result = "correct"
            match_round = "ocr"
            correct += 1
        elif matched:
            # 第2轮: spine matched_name vs GT matched_name (兜底)
            best_score2 = 0.0
            best_idx2 = -1
            for i, (orig, mt) in enumerate(gt_pool):
                if mt and matched:
                    score = _eval_match_score(matched, mt)
                    if score > best_score2:
                        best_score2 = score
                        best_idx2 = i
                        gt_display_name = mt + " (馆藏)"
                        gt_catalog_name = mt
            if best_score2 >= FUZZY_EVAL_THRESHOLD and best_idx2 >= 0:
                gt_pool.pop(best_idx2)
                result = "correct"
                match_round = "catalog"
                correct += 1
                best_score = best_score2
            else:
                extra += 1
        else:
            extra += 1

        per_spine.append({
            "spine_idx": spine_idx,
            "ocr_name": book_name,
            "matched_name": matched,
            "gt_name": gt_display_name,
            "gt_matched": gt_catalog_name,
            "gt_score": round(best_score, 4),
            "match_round": match_round,
            "result": result,
        })

    missed = len(gt_pool)
    total_gt = sum(b.get("count", 1) for b in gt_books)
    total_spine = len(spine_results)
    skipped = sum(1 for ps in per_spine if ps["result"] == "skipped")
    total_active = total_spine - skipped
    accuracy = correct / total_gt if total_gt > 0 else 0.0

    # 计数准确率: 对比每本书的 OCR count vs GT count
    # 按 matched_name 聚合 spine 的 count（来自 book_labels 的数据）
    spine_counts: dict[str, int] = {}
    for ps in per_spine:
        if ps["result"] == "correct" and ps.get("gt_matched"):
            spine_counts[ps["gt_matched"]] = spine_counts.get(ps["gt_matched"], 0) + 1

    gt_counts: dict[str, int] = {}
    unique_titles = 0
    for b in gt_books:
        mt = b.get("matched_name")
        if mt:
            gt_counts[mt] = gt_counts.get(mt, 0) + b.get("count", 1)
    unique_titles = len(gt_counts)

    count_correct = 0
    for title, gt_cnt in gt_counts.items():
        if spine_counts.get(title, 0) == gt_cnt:
            count_correct += 1

    counting_accuracy = count_correct / unique_titles if unique_titles > 0 else 0.0
    precision = correct / (correct + extra) if (correct + extra) > 0 else 0.0
    recall = correct / (correct + missed) if (correct + missed) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "per_spine": per_spine,
        "summary": {
            "correct": correct,
            "missed": missed,
            "extra": extra,
            "skipped": skipped,
            "total_gt": total_gt,
            "total_spine": total_spine,
            "total_active": total_active,
            "accuracy": round(accuracy, 4),
            "counting_accuracy": round(counting_accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "unique_titles": unique_titles,
            "count_correct": count_correct,
        },
    }


if __name__ == "__main__":
    main()
