"""根据Labelme JSON标注，将每本书脊单独裁剪出来"""
import json
import os
import cv2
import numpy as np

JSON_PATH = "book_labels/2f188063c1d9560044dd1676e9998ea8.json"
IMG_PATH = "book_images/2f188063c1d9560044dd1676e9998ea8.jpg"
OUT_DIR = "spines_from_labelme"
os.makedirs(OUT_DIR, exist_ok=True)

with open(JSON_PATH, encoding="utf-8") as f:
    data = json.load(f)

img = cv2.imread(IMG_PATH)
h, w = img.shape[:2]
print(f"Image: {w}x{h}, spines: {len(data['shapes'])}")

for i, shape in enumerate(data["shapes"]):
    pts = np.array(shape["points"], dtype=np.float32)
    label = shape.get("label", "")

    # bounding box from polygon
    x1 = max(0, int(pts[:, 0].min()))
    y1 = max(0, int(pts[:, 1].min()))
    x2 = min(w, int(pts[:, 0].max()))
    y2 = min(h, int(pts[:, 1].max()))

    crop = img[y1:y2, x1:x2]
    out_name = f"{OUT_DIR}/spine_{i:03d}_{label}.jpg"
    cv2.imwrite(out_name, crop)
    print(f"  {i:03d}: ({x1},{y1})-({x2},{y2}) {x2-x1}x{y2-y1}")

print(f"\nDone: {len(data['shapes'])} spines -> {OUT_DIR}")
