"""
OCR 管线：调用 SiliconFlow API (Qwen3-VL) 对原始书架照片进行书名识别和数量标注。

流程:
    1. 加载原始书架照片 (data/raw/{id}.jpg)
    2. 调用多模态 API 识别所有书脊的书名和数量
    3. 将结果保存为 JSON (data/ocr_results/{id}.json)

配置:
    复制 .env.example 为 .env，填入 API Key：
        cp .env.example .env
        # 编辑 .env 填入 SILICONFLOW_API_KEY

用法:
    python -m scripts.ocr.qwen_pipeline                          # 批量识别全部
    python -m scripts.ocr.qwen_pipeline --start 1 --end 10      # 指定范围
    python -m scripts.ocr.qwen_pipeline --photo_id 5             # 单张识别
    python -m scripts.ocr.qwen_pipeline --retry-failed           # 重试失败的
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from scripts.utils.paths import RAW_DIR, OCR_RESULTS_DIR, PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
MODEL = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen3-VL-32B-Instruct")

SYSTEM_PROMPT = """你是一个专业的图书盘点助手。你的任务是识别书架照片中所有书籍的书名，并统计每本书出现的册数。

请严格按照以下 JSON 格式输出，不要输出任何其他内容：
[
  {"book_name": "书名", "count": 册数},
  ...
]

注意事项：
1. 仔细识别每一本书的书名，尽量完整准确地记录
2. 同一本书出现多次时，count 填写总册数
3. 书名模糊不清的也要尽量识别，实在看不清的标注为 "未识别_位置描述"
4. 按从左到右、从上到下的顺序识别"""

MAX_RETRIES = 3
RETRY_DELAY = 5


def get_raw_image(photo_id: int) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        p = RAW_DIR / f"{photo_id}{ext}"
        if p.exists():
            return p
    return None


def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def call_ocr_api(image_path: Path) -> list[dict]:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=180, max_retries=0)

    b64 = encode_image(image_path)
    ext = image_path.suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                },
                {
                    "type": "text",
                    "text": "请识别这张书架照片中所有书籍的书名和数量。",
                },
            ],
        },
    ]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from API")

            result = json.loads(content)
            if isinstance(result, list):
                return result
            if isinstance(result, dict) and "books" in result:
                return result["books"]
            if isinstance(result, dict):
                for key in ("book_name", "name", "title"):
                    if key in result:
                        return [result]
                raise ValueError(f"Unexpected JSON structure: {list(result.keys())}")

        except json.JSONDecodeError:
            print(f"    [warn] JSON parse error, attempt {attempt + 1}/{MAX_RETRIES}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            continue
        except Exception as e:
            err_msg = str(e)
            if "timeout" in err_msg.lower() or "timed out" in err_msg.lower():
                print(f"    [timeout] {err_msg[:80]}")
                return []
            if "429" in err_msg or "rate" in err_msg.lower():
                wait = RETRY_DELAY * (attempt + 2)
                print(f"    [rate limit] retry in {wait}s...")
                time.sleep(wait)
                continue
            if attempt < MAX_RETRIES - 1:
                print(f"    [error] {err_msg[:80]}, retrying...")
                time.sleep(RETRY_DELAY)
                continue
            raise

    return []


def ocr_photo(photo_id: int) -> dict:
    image_path = get_raw_image(photo_id)
    if image_path is None:
        return {"photo_id": photo_id, "books": [], "error": "image not found"}

    books = call_ocr_api(image_path)
    return {
        "photo_id": photo_id,
        "image": image_path.name,
        "books": books,
    }


def save_ocr_result(photo_id: int, result: dict) -> Path:
    OCR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OCR_RESULTS_DIR / f"{photo_id}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="OCR 书脊文字识别（SiliconFlow Qwen3-VL）")
    parser.add_argument("--photo_id", type=int, help="指定单张图片 ID")
    parser.add_argument("--start", type=int, default=1, help="起始图片 ID")
    parser.add_argument("--end", type=int, default=None, help="结束图片 ID")
    parser.add_argument("--retry-failed", action="store_true", help="重试空结果的图片")
    args = parser.parse_args()

    if not API_KEY:
        print("Error: SILICONFLOW_API_KEY not set.")
        print("  cp .env.example .env")
        print("  # edit .env and fill in your API key")
        sys.exit(1)

    print(f"Model: {MODEL}")
    print(f"Output: {OCR_RESULTS_DIR}/\n")

    if args.photo_id:
        result = ocr_photo(args.photo_id)
        out = save_ocr_result(args.photo_id, result)
        print(f"Photo {args.photo_id}: {len(result['books'])} books -> {out}")
        return

    images = sorted(
        [f for f in RAW_DIR.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda f: int(f.stem),
    )
    if not images:
        print(f"No images in {RAW_DIR}/")
        return

    end_idx = args.end if args.end else len(images)
    end_idx = min(end_idx, len(images))

    total_books = 0
    skipped = 0
    failed = 0

    for i in range(args.start - 1, end_idx):
        photo_id = i + 1
        json_path = OCR_RESULTS_DIR / f"{photo_id}.json"

        # 断点续传：跳过已有结果
        if json_path.exists() and not args.retry_failed:
            existing = json.loads(json_path.read_text(encoding="utf-8"))
            if existing.get("books"):
                skipped += 1
                print(f"[{photo_id}/{end_idx}] SKIP ({len(existing['books'])} books)")
                continue

        print(f"[{photo_id}/{end_idx}] ", end="", flush=True)
        result = ocr_photo(photo_id)

        if "error" in result and not result["books"]:
            print(f"ERROR: {result['error']}")
            failed += 1
            save_ocr_result(photo_id, result)
            continue

        out = save_ocr_result(photo_id, result)
        n = len(result["books"])
        total_books += n
        book_names = [b.get("book_name", "?") for b in result["books"][:5]]
        preview = ", ".join(book_names)
        if n > 5:
            preview += f" ... ({n} total)"
        print(f"OK ({n} books: {preview})")

    print(f"\nDone: {end_idx - args.start + 1 - skipped - failed} annotated, {skipped} skipped, {failed} failed")
    print(f"Total books found: {total_books}")


if __name__ == "__main__":
    main()