#!/bin/bash
cd /home/sunrise/UDP_Sender/out
nohup ./UDP_Sender > log.log 2>&1 &
