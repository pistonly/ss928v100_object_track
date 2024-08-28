import cv2
import numpy as np

# 读取图像
image_path = "../data/dog1_1024_683.jpg"
img = cv2.imread(image_path)

# 选择ROI
roi = cv2.selectROI("Select ROI", img, fromCenter=False, showCrosshair=True)

# 裁剪出ROI区域
roi_img = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

# 调整ROI大小
roi_img = cv2.resize(roi_img, (96, 96))

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
roi_yuv = bgr2yuv420sp(roi_img)

# 将YUV数据保存到文件中
roi_yuv.tofile("../data/template.bin")

# 计算目标图像的中心位置和边界
l, t, w, h = roi
x = l + w / 2
y = t + h / 2
x0 = max(int(x - w), 0)
y0 = max(int(y - h), 0)
x1 = min(int(x0 + w * 2), img.shape[1])
y1 = min(int(y0 + h * 2), img.shape[0])

# 裁剪出目标图像并调整大小
target_img = img[y0:y1, x0:x1]
target_img = cv2.resize(target_img, (192, 192))

# 转换目标图像为YUV格式并保存
target_yuv = bgr2yuv420sp(target_img)
target_yuv.tofile("../data/target.bin")

# 释放窗口
cv2.destroyAllWindows()
