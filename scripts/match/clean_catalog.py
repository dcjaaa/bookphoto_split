"""
清洗馆藏目录 titles1.json → titles.json。

处理:
    1. HTML 实体反转义 (html.unescape): &quot; → "
    2. 去除首尾空白
    3. 去重 (保留首次出现)
    4. 去除空字符串

输入: data/catalog/titles1.json (原始未清洗)
输出: data/catalog/titles.json (清洗后，供匹配使用)

用法:
    python -m scripts.match.clean_catalog
"""

import html
import json
import sys
from pathlib import Path

from scripts.utils.paths import CATALOG_FILE, CATALOG_RAW_FILE

INPUT_FILE = CATALOG_RAW_FILE
OUTPUT_FILE = CATALOG_FILE


def clean_titles(titles: list[str]) -> list[str]:
    seen = set()
    cleaned = []
    for t in titles:
        t = html.unescape(t).strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(t)
    return cleaned


def main():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found")
        sys.exit(1)

    titles = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    print(f"原始: {len(titles)} 条")

    cleaned = clean_titles(titles)
    print(f"清洗后: {len(cleaned)} 条 (去除 {len(titles) - len(cleaned)} 条)")

    html_count = sum(1 for t in titles if "&" in t and ";" in t)
    empty_count = sum(1 for t in titles if not t.strip())
    from collections import Counter
    lower_counts = Counter(t.strip().lower() for t in titles)
    dup_count = sum(1 for c in lower_counts.values() if c > 1)
    print(f"  HTML 实体: {html_count} 条")
    print(f"  空字符串: {empty_count} 条")
    print(f"  重复(忽略大小写): {dup_count} 组")

    OUTPUT_FILE.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
