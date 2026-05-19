# pcb-defect-yolo26-openvino
Real-time PCB defect detection using YOLO26 optimized with OpenVINO INT8 quantization

## Overview
PCB (Printed Circuit Board) defect detection is a critical step in electronics manufacturing. Traditional manual inspection is time-consuming, inconsistent, and prone to human error at scale.

This project builds an end-to-end automated PCB defect detection pipeline using YOLO26, the latest NMS-free real-time object detection model by Ultralytics. The model is optimized using OpenVINO INT8 quantization for faster CPU inference, making it suitable for edge deployment in real manufacturing environments.

The pipeline covers the full workflow - from dataset preparation and model training to optimized inference across three modes: static image, video, and live camera feed.

### Key highlights:
- 6 defect classes: mouse_bite, spurious_copper, spur, missing_hole, open_circuit, short
- INT8 quantization via OpenVINO for faster CPU inference
- Sliding window support for high-resolution images
- Supports image, video, and live webcam inference

## Classes
This model detects 6 types of PCB defects:
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
   <img width="814" height="203" alt="dataset in Platform" src="https://github.com/user-attachments/assets/e4d1052d-6042-4e10-9fdf-d12eaa09e12a" />
   - Source dataset in Pascal VOC format
   - Converted annotations to YOLO format
   - Created data.yaml with class definitions
   - Uploaded to Ultralytics Platform for training
  
3. Training Details
   <img width="1184" height="391" alt="Overview of YOLO platform" src="https://github.com/user-attachments/assets/eb636980-400f-4ab4-a9ec-ba16344a22b3" />
   
   |Model|YOLO26n|
   |-----|-------|
   |Platform|Ultralytics Platform|
   |CPU|AMD EPYC 9655 96-Core Processor|
   |GPU|NVIDIA RTX 2000 Ada|
   |Training Cost| $0.13USD|

   
4. Export & Optimization
   <img width="1299" height="192" alt="Snipaste_2026-05-20_02-40-35-removebg-preview" src="https://github.com/user-attachments/assets/32f4f12c-dfd7-4f3c-80d4-8365339dac1a" />

   - Exported trained PyTorch model to OpenVINO format
   - Applied INT8 quantization for CPU optimization

  
## Results
|Metric|Score|
|------|------|
|mAP50|99.0%|
|mAP50-95|65.6%|
|Precision|97.3%|
|Recall|98%|

<img width="471" height="225" alt="inference improved from pytorch to openvino" src="https://github.com/user-attachments/assets/ba6de018-51a3-4a9a-b112-a07310a7ae9f" />

|Model|FPS|
|----|---|
|PyTorch|58ms|
|OpenVINO INT8|38ms|

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
