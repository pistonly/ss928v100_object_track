#!/bin/sh
export LD_LIBRARY_PATH=/mnt/data/npu:$LD_LIBRARY_PATH
sleep 10
cd /mnt/data/one_camera_track/out
nohup ./one_camera_yolo_track_2chns_1080p >> /mnt/data/track_log.log 2>&1 &
