# === IMPORTS ===
import cv2
import torch
import numpy as np
import os
import csv
import json
from pathlib import Path
from ultralytics import YOLO

# === CUSTOM SETTINGS ===
model_weights_path = r"D:/ComputerEngineering/Fawry/Fawry-Competition-final/Tracking/custom_yolov11m_640_16.pt"
model = YOLO(model_weights_path)
device = torch.device('cpu')
model.to(device)

# === INPUT/OUTPUT SETUP ===
image_dir = r"D:/ComputerEngineering/Fawry/Fawry-Competition-final/surveillance-for-retail-stores/tracking/test/01/img1"
image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
if not image_files:
    raise ValueError("Error: No images found in the directory.")

output_video_path = "output_videov11_detection_only.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
first_frame = cv2.imread(os.path.join(image_dir, image_files[0]))
if first_frame is None:
    raise ValueError("Error: Could not load the first image.")
frame_height, frame_width = first_frame.shape[:2]
out = cv2.VideoWriter(output_video_path, fourcc, 25.0, (frame_width, frame_height))

# === PROCESS FRAMES ===
frame_counter = 0
for frame_idx, image_file in enumerate(image_files):
    frame_counter += 1
    if frame_counter > 150:
        break
    frame_path = os.path.join(image_dir, image_file)
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Error: Could not load {frame_path}. Skipping...")
        continue

    print("Processing frame:", frame_path)

    results = model(frame, verbose=False)
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
    else:
        boxes = np.empty((0, 4))
        confidences = np.empty((0,))

    # Filter detections (confidence >= 0.2)
    keep = confidences >= 0.2
    boxes = boxes[keep]
    confidences = confidences[keep]

    # Draw boxes
    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write frame to output
    out.write(frame)

# === CLEANUP ===
out.release()
print(f"Detection-only video saved to {output_video_path}")
