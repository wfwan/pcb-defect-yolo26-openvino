# pcb-defect-yolo26-openvino
Real-time PCB defect detection using YOLO26 optimized with OpenVINO

## Overview
Automated PCB defect detection using YOLO26. The model is optimized using OpenVINO for faster CPU inference.
This project covers the full flow - from dataset preparation and model training to optimized inference across images and videos.

## Classes
Six types of PCB defects:
|Class|Description|
|-----|-----------|
|mouse_bite|Small notches on the edge of the PCB|
|spurious_copper|Unwanted copper remaining after etching|
|spur|Thin copper spike protruding from a trace|
|missing_hole|Drill hole that was not made|
|open_circuit|Broken trace causing disconnection|
|short|Unintended connection between two conductors|

## Pipeline
1. Dataset Preparation
   - Source dataset in Pascal VOC format and converted annotations to YOLO format
   - Created data.yaml with class definitions and uploaded to Ultralytics Platform for training
   <img width="814" height="203" alt="dataset in Platform" src="https://github.com/user-attachments/assets/e4d1052d-6042-4e10-9fdf-d12eaa09e12a" />

  
3. Training Details   
   |Model|YOLO26n|
   |-----|-------|
   |Platform|Ultralytics Platform|
   |CPU|AMD EPYC 9655 96-Core Processor|
   |GPU|NVIDIA RTX 2000 Ada|
   |Training Cost| $0.13USD|
   <img width="1184" height="391" alt="Overview of YOLO platform" src="https://github.com/user-attachments/assets/eb636980-400f-4ab4-a9ec-ba16344a22b3" />

   
5. Export & Optimization
   - Exported trained PyTorch model to OpenVINO format
<img width="1299" height="192" alt="Snipaste_2026-05-20_02-40-35-removebg-preview" src="https://github.com/user-attachments/assets/38ee65a6-18ca-430f-8fe0-4a7a0c9dffdf" />

## Results
|Metric|Score|
|------|------|
|mAP50|99.0%|
|mAP50-95|65.6%|
|Precision|97.3%|
|Recall|98%|

<img width="590" height="112" alt="Snipaste_2026-05-20_03-20-23" src="https://github.com/user-attachments/assets/06ba7ab1-9ac2-4f58-9d1b-7e54713d1531" />

## Inference Modes
### Image
```
python inference.py <path_to_image>
```
<img width="894" height="452" alt="before vs after" src="https://github.com/user-attachments/assets/ef5033b0-89ac-4a2e-9412-dd1d7d4ac931" />

### Video or Live Webcam
```
python inference_video.py <path_to_video or 0 for webcam>
```
