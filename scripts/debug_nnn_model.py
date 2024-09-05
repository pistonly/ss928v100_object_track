import numpy as np
import cv2
import re
from pathlib import Path
import pandas as pd

def read_yuv(yuv_path, imgW, imgH):
    with open(yuv_path, "rb") as f:
        data = f.read()
        img = np.ndarray((int(imgH * 1.5), imgW), dtype=np.uint8, buffer=data)
        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV12)
    return img

def get_target_region(target_path: Path):
    match = re.search(r'(\d+)-(\d+)-(\d+)-(\d+)_(\d+)', target_path.name)
    x0, y0, x1, y1, sz = map(int, match.groups())
    return x0, y0, x1, y1

def parse_result(result_path):
    with open(result_path, "r") as f:
        lines = f.readlines()
        lines = [line_.split(",") for line_ in lines]
        lines_dict = {}
        for line_ in lines:
            key = int(line_[0].split(":")[-1])
            xywh = map(float, line_[1:])
            xyxy = np.array(list(xywh))
            xyxy[2:] = xyxy[:2] + xyxy[2:]
            xyxy = map(int, xyxy)
            lines_dict[key] = xyxy
    return lines_dict


out_dir = Path("../out/")
target_files = {f.name.split("_")[1]:f for f in out_dir.iterdir() if f.name.startswith("target_")}
result_path = "../out/results.csv"
result = parse_result(result_path)

cv2.namedWindow("debug", cv2.WINDOW_NORMAL)

for i in range(1, 104):
    img_path = f"../out/yuv_{i}.bin"
    img = read_yuv(img_path, 3840, 2160)

    target_path = target_files[f"{i}"]
    target_x0, target_y0, target_x1, target_y1 = get_target_region(target_path)

    roi_x0, roi_y0, roi_x1, roi_y1 = result[i]

    cv2.rectangle(img, (target_x0, target_y0), (target_x1, target_y1), (0, 255, 0), 2)
    cv2.rectangle(img,  (roi_x0, roi_y0), (roi_x1, roi_y1), (0, 0, 255), 2)
    
    cv2.imshow("debug", img)
    key = cv2.waitKey(-1)
    if key == ord("n"):
        continue

    if key == 27:
        break




# template_bin = "../out/template_1_1568-1442-1608-1482_48.bin"

# # 使用正则表达式匹配4个整数

# x0, y0, x1, y1, sz = map(int, match.groups())

# with open(template_bin, "rb") as f:
#     data = f.read()
#     template_out = np.ndarray((int(sz * 1.5), sz), dtype=np.uint8, buffer=data)
#     template_bgr = cv2.cvtColor(template_out, cv2.COLOR_YUV2BGR_NV12)
# cv2.imwrite("template_bgr.jpg", template_bgr)




