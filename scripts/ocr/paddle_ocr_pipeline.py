"""
PaddleOCR 管线：通过 subprocess 调用 .venv-paddle 下的 PaddleOCR

使用方法（在主 .venv 下运行）:
    python -m scripts.ocr.paddle_ocr_pipeline --start 1 --end 10
    python -m scripts.ocr.paddle_ocr_pipeline --photo_id 5
    python -m scripts.ocr.paddle_ocr_pipeline --batch
    python -m scripts.ocr.paddle_ocr_pipeline --retry-failed
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils.paths import SPLIT_DIR, DATA_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PADDLE_VENV_PYTHON = PROJECT_ROOT / ".venv-paddle" / "bin" / "python"
PADDLE_OCR_CLI = PROJECT_ROOT / "scripts" / "ocr" / "paddle_ocr_cli.py"

PADDLE_OCR_RESULTS_DIR = DATA_DIR / "paddle_ocr_results"


def run_paddle_ocr_single(image_path: str, no_rotate: bool = False, use_gpu: bool = True) -> dict:
    cmd = [str(PADDLE_VENV_PYTHON), str(PADDLE_OCR_CLI), "--image", image_path]
    if no_rotate:
        cmd.append("--no-rotate")
    if not use_gpu:
        cmd.append("--no-gpu")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, encoding="utf-8")
        if result.returncode != 0:
            return {"error": result.stderr[:500], "text": "", "lines": []}
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return {"error": "OCR timeout (120s)", "text": "", "lines": []}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "text": "", "lines": []}
    except Exception as e:
        return {"error": str(e), "text": "", "lines": []}


def run_paddle_ocr_batch(input_dir: str, output_dir: str, no_rotate: bool = False, use_gpu: bool = True):
    cmd = [
        str(PADDLE_VENV_PYTHON),
        str(PADDLE_OCR_CLI),
        "--input", input_dir,
        "--output", output_dir,
    ]
    if no_rotate:
        cmd.append("--no-rotate")
    if not use_gpu:
        cmd.append("--no-gpu")

    subprocess.run(cmd, timeout=3600)


def process_photo(photo_id: int, use_gpu: bool = True) -> dict:
    spine_dir = SPLIT_DIR / str(photo_id)
    if not spine_dir.exists():
        return {"photo_id": photo_id, "spines": [], "error": "spine directory not found"}

    spine_images = sorted(spine_dir.glob("spine_*.png"))
    if not spine_images:
        return {"photo_id": photo_id, "spines": [], "error": "no spine images found"}

    spines = []
    for spine_img in spine_images:
        result = run_paddle_ocr_single(str(spine_img), use_gpu=use_gpu)
        spine_result = {
            "spine_file": spine_img.name,
            "text": result.get("text", ""),
            "lines": result.get("lines", []),
            "error": result.get("error", ""),
        }
        if not spine_result["error"]:
            spines.append(spine_result)

    return {"photo_id": photo_id, "spines": spines}


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR pipeline for book spine text recognition")
    parser.add_argument("--photo_id", type=int, help="Process a single photo's spines")
    parser.add_argument("--start", type=int, default=1, help="Start photo ID")
    parser.add_argument("--end", type=int, default=None, help="End photo ID")
    parser.add_argument("--batch", action="store_true", help="Batch process split directory via PaddleOCR CLI")
    parser.add_argument("--retry-failed", action="store_true", help="Retry photos with empty results")
    parser.add_argument("--no-gpu", action="store_true", help="Use CPU only")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for batch mode")
    args = parser.parse_args()

    use_gpu = not args.no_gpu

    if not PADDLE_VENV_PYTHON.exists():
        print(f"Error: PaddleOCR venv not found at {PADDLE_VENV_PYTHON}")
        print("Run: uv venv .venv-paddle --python 3.13")
        sys.exit(1)

    if args.batch:
        output_dir = args.output_dir or str(PADDLE_OCR_RESULTS_DIR / "split")
        print(f"Batch OCR: {SPLIT_DIR} -> {output_dir}")
        run_paddle_ocr_batch(str(SPLIT_DIR), output_dir, use_gpu=use_gpu)
        return

    PADDLE_OCR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.photo_id:
        result = process_photo(args.photo_id, use_gpu=use_gpu)
        out_file = PADDLE_OCR_RESULTS_DIR / f"{args.photo_id}.json"
        out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        n_text = len([s for s in result["spines"] if s.get("text")])
        print(f"Photo {args.photo_id}: {n_text}/{len(result['spines'])} spines with text -> {out_file}")
        return

    split_dirs = sorted([d for d in SPLIT_DIR.iterdir() if d.is_dir()], key=lambda d: int(d.name))
    if not split_dirs:
        print(f"No split directories found in {SPLIT_DIR}")
        return

    end_idx = args.end if args.end else len(split_dirs)
    end_idx = min(end_idx, len(split_dirs))

    total_text = 0
    skipped = 0
    failed = 0

    for i in range(args.start - 1, end_idx):
        photo_id = int(split_dirs[i].name)
        out_file = PADDLE_OCR_RESULTS_DIR / f"{photo_id}.json"

        if out_file.exists() and not args.retry_failed:
            existing = json.loads(out_file.read_text(encoding="utf-8"))
            if existing.get("spines"):
                skipped += 1
                n = len([s for s in existing["spines"] if s.get("text")])
                print(f"[{photo_id}] SKIP ({n} spines with text)")
                continue

        print(f"[{photo_id}] ", end="", flush=True)
        result = process_photo(photo_id, use_gpu=use_gpu)

        if result.get("error") and not result.get("spines"):
            print(f"ERROR: {result['error']}")
            failed += 1
            out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            continue

        n_text = len([s for s in result["spines"] if s.get("text")])
        total_text += n_text
        out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"OK ({n_text}/{len(result['spines'])} spines with text)")

    print(f"\nDone: {end_idx - args.start + 1 - skipped - failed} processed, {skipped} skipped, {failed} failed")
    print(f"Total spines with text: {total_text}")


if __name__ == "__main__":
    main()