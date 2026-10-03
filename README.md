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
  
2. Training Details   
   |Model|YOLO26n|
   |-----|-------|
   |Platform|Ultralytics Platform|
   |CPU|AMD EPYC 9655 96-Core Processor|
   |GPU|NVIDIA RTX 2000 Ada|
   |Training Cost| $0.13USD|
   
3. Export & Optimization
   - Exported trained PyTorch model to OpenVINO format
     
## Results
|Metric|Score|
|------|------|
|mAP50|99.0%|
|mAP50-95|65.6%|
|Precision|97.3%|
|Recall|98%|

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
