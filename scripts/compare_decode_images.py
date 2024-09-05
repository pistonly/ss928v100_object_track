import numpy as np
from pathlib import Path
import cv2

image_dir = Path("/home/liuyang/Documents/SOT/7-one_camera_track/out/")
image_paths = [str(f) for f in image_dir.iterdir() if f.name.startswith("frame_") and f.name.endswith(".bin")]
img_h = 2160
img_w = 3840

for img_path in image_paths:
    with open(img_path, "rb") as f:
        yuv = f.read()
        yuv = np.ndarray((int(img_h * 1.5), img_w), dtype=np.uint8, buffer=yuv)
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        bgr_path = Path(img_path).with_suffix(".jpg")
        cv2.imwrite(str(bgr_path), bgr)

