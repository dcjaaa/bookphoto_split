"""
HTTP API client: typed wrappers around the FastAPI endpoints.

Every public function returns a dict (JSON body) or raises an
exception with a user-friendly message (no httpx leakage).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE = "http://127.0.0.1:8000"


class ClientError(Exception):
    """Wrapper for any client/transport error."""


# ---------------------------------------------------------------------------
# public helpers
# ---------------------------------------------------------------------------

def _url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------

def health(base_url: str = DEFAULT_BASE, timeout: float = 5.0) -> dict[str, Any]:
    """GET /api/health → dict."""
    try:
        r = httpx.get(_url(base_url, "/api/health"), timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise ClientError(f"health check failed ({e.response.status_code})") from e
    except Exception as e:
        raise ClientError(f"health check failed: {e}") from e


def segment(
    image_path: Path,
    base_url: str = DEFAULT_BASE,
    conf: float = 0.25,
    imgsz: int = 960,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """POST /api/segment — spine detection. Returns {image_size, count, boxes}."""
    params = {"conf": conf, "imgsz": imgsz}
    try:
        with open(image_path, "rb") as fh:
            r = httpx.post(
                _url(base_url, "/api/segment"),
                params=params,
                files={"file": (image_path.name, fh, "image/jpeg")},
                timeout=timeout,
            )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        raise ClientError(f"segment failed ({e.response.status_code}): {detail}") from e
    except Exception as e:
        raise ClientError(f"segment failed: {e}") from e


def ocr_spine(
    image_path: Path,
    base_url: str = DEFAULT_BASE,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """POST /api/ocr/spine — 单脊 OCR + 馆藏匹配。返回 {book_name, matched_name, score, strategy, needs_review}。"""
    try:
        with open(image_path, "rb") as fh:
            r = httpx.post(
                _url(base_url, "/api/ocr/spine"),
                files={"file": (image_path.name, fh, "image/png")},
                timeout=timeout,
            )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        raise ClientError(f"OCR spine failed ({e.response.status_code}): {detail}") from e
    except Exception as e:
        raise ClientError(f"OCR spine failed: {e}") from e


def inventory(
    results: list[dict[str, Any]],
    base_url: str = DEFAULT_BASE,
    threshold: float = 0.7,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """POST /api/inventory — catalog matching. Returns {book_counts, match_log, ...}."""
    try:
        r = httpx.post(
            _url(base_url, "/api/inventory"),
            json={"results": results, "threshold": threshold},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        raise ClientError(f"inventory failed ({e.response.status_code}): {detail}") from e
    except Exception as e:
        raise ClientError(f"inventory failed: {e}") from e


def inventory_all(
    base_url: str = DEFAULT_BASE,
    threshold: float = 0.6,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """GET /api/inventory/all — full inventory summary."""
    try:
        r = httpx.get(
            _url(base_url, "/api/inventory/all"),
            params={"threshold": threshold},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        raise ClientError(f"inventory/all failed ({e.response.status_code}): {detail}") from e
    except Exception as e:
        raise ClientError(f"inventory/all failed: {e}") from e
