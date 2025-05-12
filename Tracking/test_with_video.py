import cv2
import torch
import numpy as np
import os
import csv
import json
import matplotlib.pyplot as plt
from pathlib import Path
import torchvision
import torchvision.ops  # explicitly import ops
_ = torchvision.extension  # force the extension to load
from ultralytics import YOLO
from boxmot import BotSort
from boxmot.appearance.backbones.osnet import osnet_x0_25

# ======== CUSTOM SETTINGS ========

# (1) Custom YOLOv8 model weights
model_weights_path = r"D:/ComputerEngineering/Fawry/Fawry-Competition-final/Tracking/custom_yolov11_retail_best_imgsz.pt"
model = YOLO(model_weights_path)
device = torch.device('cpu')
model.to(device)

# (2) Custom OSNet weights (.pth)
custom_osnet_path = Path(r"D:/ComputerEngineering/Fawry/Fawry-Competition-final/Tracking/osnet_x0_25_market1501_finetuned.pt")

# ======== INPUT/OUTPUT SETUP (VIDEO VERSION) ========
video_input_path = r"D:\ComputerEngineering\Fawry\Fawry-Competition-final\archive\D8_S20250304210514_E20250304210515.mp4"
cap = cv2.VideoCapture(video_input_path)
if not cap.isOpened():
    raise ValueError(f"Error: Could not open video {video_input_path}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Output video 2× faster (fps * 2)
output_video_path = "D8_S20250304210514_E20250304210515.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps * 2, (frame_width, frame_height))
print("Video writer dimensions:", (frame_width, frame_height))

# Output CSV
output_csv = "D8_S20250304210514_E20250304210515.csv"
csv_file = open(output_csv, "w", newline='')
fieldnames = ["ID", "frame", "objects", "objective"]
writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
writer.writeheader()

# ======== INITIALIZE TRACKER ========
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tracker = BotSort(
    reid_weights=custom_osnet_path,
    device=device,
    half=False,
    frame_rate=fps,
)

# ======== PROCESS VIDEO FRAMES ========
frame_counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    print(f"Processing frame {frame_counter}/{frame_count} with shape: {frame.shape}")

    # --- Detection with YOLO ---
    results = model(frame, verbose=False)
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
    else:
        boxes = np.empty((0, 4))
        confidences = np.empty((0,))

    # Filter detections (confidence threshold >= 0.0)
    if confidences.size > 0:
        keep = confidences >= 0.0
        boxes = boxes[keep]
        confidences = confidences[keep]
    else:
        boxes = np.empty((0, 4))
        confidences = np.empty((0,))

    # Prepare detections
    if boxes.shape[0] > 0:
        dummy_class = np.zeros((boxes.shape[0], 1))
        scores = confidences.reshape(-1, 1)
        dets = np.hstack([boxes, scores, dummy_class])
    else:
        dets = np.empty((0, 6))

    # Update tracker
    res = tracker.update(dets, frame)

    # Annotate
    tracker.plot_results(frame, show_trajectories=True)
    annotated_frame = frame.copy()

    # Write tracking data to CSV
    objects_list = []
    if res is not None and res.shape[0] > 0:
        for det in res:
            x1, y1, x2, y2, track_id, conf, *_ = det
            width = x2 - x1
            height = y2 - y1
            obj = {
                'tracked_id': int(track_id),
                'x': float(x1),
                'y': float(y1),
                'w': float(width),
                'h': float(height),
                'confidence': conf
            }
            objects_list.append(obj)
    objects_str = json.dumps(objects_list)
    writer.writerow({
        "ID": frame_counter,
        "frame": float(frame_counter),
        "objects": objects_str,
        "objective": "tracking"
    })

    # Write to output video
    out.write(annotated_frame)

# ======== RELEASE RESOURCES ========
cap.release()
csv_file.close()
out.release()
print(f"Tracking results saved to {output_csv}")
print(f"Output video saved to {output_video_path}")
