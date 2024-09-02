import cv2
import numpy as np

# 读取图像
image_path = "../data/frame00897.jpg"
img = cv2.imread(image_path)

# crop: x0, y0, x1, y1
template_crop = [1568, 1442, 1608, 1482]
template_shape = [48, 48]
target_crop = [1540, 1412, 1638, 1510]
target_shape = [112, 112]


template = img[template_crop[1]:template_crop[3], template_crop[0]:template_crop[2]]
target = img[target_crop[1]:target_crop[3], target_crop[0]:target_crop[2]]

# cpp output
template_bin = "../out/template.bin"
target_bin = "../out/target.bin"

template_H, template_W = template.shape[0:2]
target_H, target_W = target.shape[0:2]
with open(template_bin, "rb") as f:
    data = f.read()
    print(len(data), template_shape[0] * template_shape[1] * 1.5)
    template_out = np.ndarray((int(template_shape[0] * 1.5), template_shape[1]), dtype=np.uint8, buffer=data)

with open(target_bin, "rb") as f:
    data = f.read()
    print(len(data), target_shape[0] * target_shape[1] * 1.5)
    target_out = np.ndarray((int(target_shape[0] * 1.5), target_shape[1]), dtype=np.uint8, buffer=data)

template_out_bgr = cv2.cvtColor(template_out, cv2.COLOR_YUV2BGR_NV12)
target_out_bgr = cv2.cvtColor(target_out, cv2.COLOR_YUV2BGR_NV12)


cv2.imwrite("template_0.jpg", template)
cv2.imwrite("template_1.jpg", template_out_bgr)

cv2.imwrite("target_0.jpg", target)
cv2.imwrite("target_1.jpg", target_out_bgr)

image_yuv_path = "../data/frame00897_1575-1453-31-21.bin"
with open(image_yuv_path, "rb") as f:
    data = f.read()
    image_yuv = np.ndarray((int(2160 * 1.5), 3840), dtype=np.uint8, buffer=data)
image_bgr = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR_NV12)
cv2.imwrite("image_bgr.jpg", image_bgr)
