"""
公共裁剪工具：按多边形 mask 裁剪书脊。

4 处裁剪逻辑的统一入口：
  - annotate 阶段 (crop_spines.py)：多边形 → mask → RGBA 透明 PNG
  - infer 阶段 (predict.py)：已有 mask → RGBA 透明 PNG
  - prepare 阶段 (to_yolo.py)：多边形 → mask → JPG（无 alpha）
  - infer 阶段 (crop_from_predict.py)：仅 bbox 矩形裁剪 → JPG（无 mask）
"""

from __future__ import annotations

import cv2
import numpy as np


def polygon_to_mask(points: list[list[float]] | np.ndarray, h: int, w: int) -> np.ndarray:
    """多边形点列表 → 二值 mask (h×w, uint8, 0/255)。"""
    pts = np.array(points, dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def clamp_bbox(bbox: tuple[int, int, int, int], h: int, w: int) -> tuple[int, int, int, int]:
    """将 bbox 裁剪到图像边界内。返回 (x1, y1, x2, y2)。"""
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return x1, y1, x2, y2


def crop_with_mask(
    img: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    alpha: bool = False,
) -> np.ndarray | None:
    """
    按 mask 裁剪书脊。

    Args:
        img:  原图 (H×W×3 BGR)
        mask: 二值 mask (H×W, 0/255)，与 img 同尺寸
        bbox: (x1, y1, x2, y2) 裁剪框
        alpha: True → 返回 RGBA (背景透明，用于 PNG)
               False → 返回 BGR (mask 外区域置黑，用于 JPG)

    Returns:
        裁剪后的图 (C×H'×W')，或 None（空 bbox）。
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, h, w)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    if alpha:
        # 背景透明 RGBA：mask 区域保留原图，其余 alpha=0
        b, g, r = cv2.split(crop)
        return cv2.merge([b, g, r, mask_crop])
    else:
        # mask 外区域置黑（JPG 无 alpha 通道）
        return cv2.bitwise_and(crop, crop, mask=mask_crop)


def crop_bbox_only(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray | None:
    """
    仅按 bbox 矩形裁剪（无 mask）。

    Returns:
        裁剪后的 BGR 图，或 None（空 bbox）。
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, h, w)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def polygon_bbox(points: list[list[float]] | np.ndarray) -> tuple[int, int, int, int]:
    """从多边形点列表计算 bbox (x1, y1, x2, y2)。"""
    pts = np.array(points, dtype=np.int32)
    x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
    x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
    return x1, y1, x2, y2
