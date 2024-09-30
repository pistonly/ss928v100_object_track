import pandas as pd
import cv2
from pathlib import Path
import time

result_path = "/home/liuyang/Documents/tmp/sot/results.csv"
rtsp_url = "rtsp://172.23.24.52:8554/test"

# columns: [imageId,  trackerId,        l ,       t,        w,       h]
track_results = pd.read_csv(result_path)
track_results.columns = [col.strip() for col in track_results.columns]

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print(f"Can't open RTSP steam: {rtsp_url}")
    exit()

save = False
cv2.namedWindow("RTSP Stream", cv2.WINDOW_NORMAL)
output_dir = Path("./tmp")
if save:
    output_dir.mkdir(exist_ok=True)

img_id = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
color_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("can't receive frame. Existing...")
        break

    # draw track bbox
    track_result_frame = track_results[track_results['imageId'] == img_id]
    for i in range(len(track_result_frame)):
        track_i = track_result_frame.iloc[i]
        track_id = int(track_i['trackerId'])
        x0 = int(track_i['l'])
        y0 = int(track_i['t'])
        w = int(track_i['w'])
        h = int(track_i['h'])
        color = colors[track_id % len(colors)]
        cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), color, 2)

    if cv2.waitKey(1) == ord("q"):
        break
    cv2.imshow("RTSP Stream", frame)
    time.sleep(0.5)
    img_id += 1

