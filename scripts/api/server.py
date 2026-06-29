"""
FastAPI 后端：把书脊分割 / OCR 识别 / 馆藏匹配 暴露为 HTTP 接口。

供前端（PyQt5 桌面端 / Web 前端）调用，前端无需安装 torch/ultralytics/openai。

启动:
    uvicorn scripts.api.server:app --reload --host 0.0.0.0 --port 8000
    python -m scripts.api.server

接口:
    POST /api/segment        上传图片 → 书脊分割结果
    POST /api/ocr            上传图片 → 书名+数量识别
    POST /api/inventory      OCR结果JSON → 馆藏匹配+去重计数
    GET  /api/ocr/{photo_id} 取已有OCR结果（按图片ID）
    GET  /api/inventory/all  汇总全部OCR结果 → 盘点统计
    GET  /api/catalog        馆藏目录书名列表
    GET  /api/health         健康检查
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from scripts.utils.paths import (
    RAW_DIR, OCR_RESULTS_DIR, CATALOG_DIR, OUTPUT_DIR,
    SEG_MODEL_PATH, CATALOG_FILE, INVENTORY_RESULT_FILE,
)

app = FastAPI(
    title="BookPhoto Split API",
    description="图书盘点后端：书脊分割、OCR识别、馆藏匹配",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_seg_model = None


def _get_seg_model():
    global _seg_model
    if _seg_model is None:
        if not SEG_MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail=f"模型权重不存在: {SEG_MODEL_PATH}")
        from ultralytics import YOLO
        _seg_model = YOLO(str(SEG_MODEL_PATH))
    return _seg_model


def _load_catalog() -> list[str]:
    if not CATALOG_FILE.exists():
        raise HTTPException(status_code=504, detail=f"馆藏目录不存在: {CATALOG_FILE}")
    return json.loads(CATALOG_FILE.read_text(encoding="utf-8"))


class OcrResult(BaseModel):
    photo_id: int | None = None
    image: str | None = None
    books: list[dict]


class InventoryRequest(BaseModel):
    results: list[OcrResult]
    threshold: float = 0.6


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_exists": SEG_MODEL_PATH.exists(),
        "catalog_exists": CATALOG_FILE.exists(),
        "ocr_results": len(list(OCR_RESULTS_DIR.glob("*.json"))) if OCR_RESULTS_DIR.exists() else 0,
    }


@app.post("/api/segment")
async def segment(file: UploadFile = File(...), conf: float = 0.25, imgsz: int = 960):
    """上传书架照片 → YOLO 书脊分割。"""
    model = _get_seg_model()
    suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        results = model.predict(source=str(tmp_path), conf=conf, imgsz=imgsz, save=False, retina_masks=False)
        r = results[0]
        masks_xy = r.masks.xy if r.masks is not None else []
        boxes = []
        if r.boxes is not None:
            for i, b in enumerate(r.boxes):
                polygon = None
                if i < len(masks_xy) and len(masks_xy[i]) >= 3:
                    pts = masks_xy[i]
                    if len(pts) > 50:
                        import cv2
                        arc = cv2.arcLength(pts.astype("int32"), True)
                        poly = cv2.approxPolyDP(pts.astype("int32"), 0.01 * arc, True).reshape(-1, 2)
                        pts = poly.astype("float32")
                    polygon = [[round(float(x), 1), round(float(y), 1)] for x, y in pts]
                boxes.append({
                    "bbox": [round(float(x), 1) for x in b.xyxy[0].tolist()],
                    "confidence": round(float(b.conf[0]), 4),
                    "label": model.names[int(b.cls[0])],
                    "polygon": polygon,
                })
        return {
            "image_size": {"width": int(r.orig_shape[1]), "height": int(r.orig_shape[0])},
            "count": len(boxes),
            "boxes": boxes,
        }
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/ocr")
async def ocr(file: UploadFile = File(...)):
    """上传书架照片 → Qwen3-VL 识别书名+数量。"""
    from scripts.ocr.qwen_pipeline import call_ocr_api

    suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        books = call_ocr_api(tmp_path)
        return {"books": books, "count": len(books)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR 失败: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/api/ocr/{photo_id}")
async def get_ocr(photo_id: int):
    """按 photo_id 取已保存的 OCR 结果。"""
    path = OCR_RESULTS_DIR / f"{photo_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"无 OCR 结果: {photo_id}")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/ocr")
async def list_ocr():
    """列出所有已处理的 OCR 结果概览。"""
    if not OCR_RESULTS_DIR.exists():
        return {"total": 0, "results": []}
    items = []
    for f in sorted(OCR_RESULTS_DIR.glob("*.json"), key=lambda x: int(x.stem)):
        d = json.loads(f.read_text(encoding="utf-8"))
        items.append({
            "photo_id": d.get("photo_id", int(f.stem)),
            "book_count": len(d.get("books", [])),
            "has_error": bool(d.get("error")),
        })
    return {"total": len(items), "results": items}


@app.post("/api/inventory")
async def inventory(req: InventoryRequest):
    """对给定 OCR 结果做馆藏匹配 + 去重计数。"""
    from scripts.match.inventory import count_books, build_catalog_index

    catalog = _load_catalog()
    index = build_catalog_index(catalog)
    results = [r.model_dump() for r in req.results]
    inv = count_books(results, index, req.threshold)
    return {
        "book_counts": inv["book_counts"],
        "match_log": inv["match_log"],
        "total_copies": sum(inv["book_counts"].values()),
        "matched": sum(1 for m in inv["match_log"] if m["matched_name"]),
        "unique_titles": len(inv["book_counts"]),
    }


@app.get("/api/inventory/all")
async def inventory_all(threshold: float = 0.6):
    """汇总全部已有 OCR 结果 → 盘点统计。"""
    from scripts.match.inventory import count_books, load_ocr_results, build_catalog_index

    results = load_ocr_results()
    if not results:
        raise HTTPException(status_code=404, detail="无 OCR 结果，请先运行 qwen_pipeline")
    catalog = _load_catalog()
    index = build_catalog_index(catalog)
    inv = count_books(results, index, threshold)
    INVENTORY_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    INVENTORY_RESULT_FILE.write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "book_counts": inv["book_counts"],
        "total_copies": sum(inv["book_counts"].values()),
        "matched": sum(1 for m in inv["match_log"] if m["matched_name"]),
        "unique_titles": len(inv["book_counts"]),
        "photos": len(results),
        "saved_to": str(INVENTORY_RESULT_FILE),
    }


@app.get("/api/catalog")
async def catalog(limit: int = 100, offset: int = 0):
    """馆藏目录书名列表（分页）。"""
    titles = _load_catalog()
    return {
        "total": len(titles),
        "limit": limit,
        "offset": offset,
        "titles": titles[offset:offset + limit],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("scripts.api.server:app", host="0.0.0.0", port=8000, reload=True)
