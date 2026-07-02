import json
import re
import time
from collections import defaultdict

import jieba

from scripts.utils.paths import CATALOG_FILE, CATALOG_DIR

# 停用词: 标点、单字
_SKIP_TAGS = {"（", "）", "(", ")", "——", "—", "、", "·", "：", ":", "《", "》", "“", "”", "，", "。", "；", "；", "！", "？"}

_OUTPUT = CATALOG_DIR / "keyword_index.json"


def build():
    catalog = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    print(f"Catalog: {len(catalog)} titles")

    t0 = time.time()
    keyword_titles: dict[str, list[int]] = defaultdict(list)

    for idx, title in enumerate(catalog):
        if not title:
            continue
        words = list(jieba.cut(title))
        seen = set()
        for w in words:
            w = w.strip()
            if not w or w in _SKIP_TAGS or len(w) <= 1:
                continue
            if re.match(r"^[\d\s\.\,\-A-Za-z]+$", w):
                # 纯英文词: ≥3字且非停用词才保留 (Visual/FoxPro/Java/Linux 都是有效关键词)
                if len(w) <= 2:
                    continue
                if w.lower() in {"the", "and", "for", "with", "from", "into", "over",
                                  "that", "this", "what", "when", "were", "will",
                                  "book", "books", "text", "guide", "paper", "press",
                                  "introduction", "edition", "edition", "manual"}:
                    continue
                # 通过检查，保留为关键词
            if w not in seen:
                seen.add(w)
                keyword_titles[w].append(idx)

        # 兜底: jieba 分词后无≥2字关键词的短书名(如"你的英文又错了!")
        # 用前3字 bigram 补救, 确保每个书名至少有一个关键词
        if not seen:
            clean = re.sub(r'[\s\d\.\,\;\:\!\?\(\)\[\]\{\}（）【】《》〈〉「」『』\"\'—…·★☆●◆\-/\\&@#\$%\^&\*\+\=~`#]+', '', title)
            for i in range(len(clean) - 1):
                bg = clean[i:i+2]
                if bg not in seen:
                    seen.add(bg)
                    keyword_titles[bg].append(idx)

    t1 = time.time()
    print(f"分词建索引: {t1 - t0:.1f}s")
    print(f"唯一关键词: {len(keyword_titles)}")

    # 统计频率
    freq = {k: len(v) for k, v in keyword_titles.items()}
    sorted_freq = sorted(freq.items(), key=lambda x: -x[1])

    # 输出紧凑格式: {keywords: {词: [title_idx, ...]}, frequencies: {词: count}}
    output = {
        "total_titles": len(catalog),
        "unique_keywords": len(keyword_titles),
        "frequencies": freq,
        "keywords": keyword_titles,
    }

    _OUTPUT.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    file_size = _OUTPUT.stat().st_size / 1024 / 1024
    print(f"保存: {_OUTPUT} ({file_size:.1f} MB)")
    print(f"\n高频词 Top 20:")
    for k, v in sorted_freq[:20]:
        print(f"  {k}: {v}")
    print(f"\n低频词(≤50次): {sum(1 for v in freq.values() if v <= 50)}")
    print(f"低频词(≤20次): {sum(1 for v in freq.values() if v <= 20)}")


if __name__ == "__main__":
    build()
