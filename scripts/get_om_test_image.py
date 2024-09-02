import cv2
import numpy as np
from pathlib import Path

# 读取图像
image_path = "../data/frame00897.jpg"
img = cv2.imread(image_path)

# 选择ROI
roi = cv2.selectROI("Select ROI", img, fromCenter=False, showCrosshair=True)

x0, y0, w, h = roi[0], roi[1], roi[2], roi[3]

def bgr2yuv420sp(bgr):
    # BGR to YUV420
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    h, w = yuv.shape
    h_plane = h // 3
    yuv_sp = yuv.copy()
    # YUV420 to YUV420sp
    uv_sp = yuv_sp[h_plane * 2:].reshape(2, h_plane // 2, w).transpose(1, 2, 0).reshape(h_plane, -1)
    yuv_sp[h_plane * 2:] = uv_sp
    return yuv_sp

# 转换为YUV格式
yuv = bgr2yuv420sp(img)

# 将YUV数据保存到文件中
yuv_file_path = Path(image_path).parent / f"{Path(image_path).stem}_{x0}-{y0}-{w}-{h}.bin"
yuv.tofile(str(yuv_file_path))

