"""
PaddleOCR CLI — 运行在 .venv-paddle (Python 3.13 + PaddlePaddle 3.0.0) 下

用法 (.venv-paddle/bin/python):
    python scripts/ocr/paddle_cli.py --image data/crops_labelme/1/spine_000.png
    python scripts/ocr/paddle_cli.py --image data/crops_labelme/1/spine_000.png --no-rotate
    python scripts/ocr/paddle_cli.py --input data/crops_labelme/1 --output data/paddle_ocr_results/1

环境变量:
    FLAGS_use_mkldnn=0       禁用 OneDNN（必须，否则崩溃）
    FLAGS_enable_pir_api=0  禁用 PIR API（必须，否则崩溃）
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

import cv2
import numpy as np
import paddle
paddle.set_flags({"FLAGS_use_mkldnn": False, "FLAGS_enable_pir_api": False})

from paddleocr import PaddleOCR

OCR_ENGINE = None


def get_ocr_engine():
    global OCR_ENGINE
    if OCR_ENGINE is None:
        OCR_ENGINE = PaddleOCR(
            lang="ch",
            ocr_version="PP-OCRv4",
            use_doc_orientation_classify=True,
            use_textline_orientation=True,
        )
    return OCR_ENGINE


def rotate_if_vertical(image: np.ndarray) -> tuple[np.ndarray, bool]:
    h, w = image.shape[:2]
    if h > w:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return rotated, True
    return image, False


def ocr_single(image_path: str, no_rotate: bool = False) -> dict:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": f"Cannot read image: {image_path}", "text": "", "lines": []}

    was_rotated = False
    if not no_rotate:
        img, was_rotated = rotate_if_vertical(img)

    ocr = get_ocr_engine()
    result = ocr.predict(img)

    lines = []
    all_text_parts = []

    if result:
        for res in result:
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            polys = res.get("rec_polys", res.get("dt_polys", []))
            for i, text in enumerate(texts):
                if not text.strip():
                    continue
                score = float(scores[i]) if i < len(scores) else 0.0
                poly = polys[i] if i < len(polys) else []
                lines.append({
                    "text": text.strip(),
                    "confidence": round(score, 4),
                    "bbox": poly.tolist() if hasattr(poly, "tolist") else poly,
                })
                all_text_parts.append(text.strip())

    full_text = "".join(all_text_parts)

    return {
        "image": str(image_path),
        "was_rotated": was_rotated,
        "text": full_text,
        "lines": lines,
    }


def batch_ocr(input_dir: str, output_dir: str, no_rotate: bool = False):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        [f for f in input_path.rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")],
        key=lambda f: str(f),
    )

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images in {input_dir}")
    _ = get_ocr_engine()

    total = len(image_files)
    for i, img_file in enumerate(image_files, 1):
        rel_path = img_file.relative_to(input_path)
        out_file = output_path / f"{img_file.stem}.json"

        if out_file.exists():
            existing = json.loads(out_file.read_text(encoding="utf-8"))
            if existing.get("text") or existing.get("lines"):
                print(f"[{i}/{total}] SKIP {rel_path}")
                continue

        print(f"[{i}/{total}] {rel_path} ... ", end="", flush=True)
        start = time.time()

        result = ocr_single(str(img_file), no_rotate=no_rotate)
        result["elapsed"] = round(time.time() - start, 2)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        text_preview = result.get("text", "")[:50]
        n_lines = len(result.get("lines", []))
        print(f"OK ({n_lines} lines, {result.get('elapsed', 0)}s) {text_preview}")

    print(f"\nDone: {total} images processed -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR CLI for book spine text recognition")
    parser.add_argument("--image", type=str, help="Single image path to OCR")
    parser.add_argument("--input", type=str, help="Input directory for batch OCR")
    parser.add_argument("--output", type=str, help="Output directory for batch results")
    parser.add_argument("--no-rotate", action="store_true", help="Disable auto-rotation of vertical text")
    args = parser.parse_args()

    if args.image:
        result = ocr_single(args.image, no_rotate=args.no_rotate)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.input:
        if not args.output:
            args.output = str(Path(args.input).parent / "paddle_ocr_results")
        batch_ocr(args.input, args.output, no_rotate=args.no_rotate)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()