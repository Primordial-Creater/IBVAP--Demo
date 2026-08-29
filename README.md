# IBVAP
## Intelligent Border Video Analytics Platform

AI-powered software-defined surveillance and video analytics platform for border security and strategic installations.

## Capabilities

- Human detection and tracking
- Vehicle detection and classification
- Face detection
- Automatic Number Plate Recognition (ANPR)
- OCR
- Virtual fence intrusion detection
- Suspicious activity detection
- Night-time movement detection
- Real-time event generation
- Event logging
- Unified AI video analytics pipeline

## AI Stack

- Python
- PyTorch
- Ultralytics YOLO
- OpenCV
- EasyOCR
- NumPy

## Hardware

NVIDIA CUDA-enabled GPU recommended.

The system has been tested with an NVIDIA GeForce RTX 3050 6GB GPU.

## Project Structure

```text
CJP--IBVAP/
│
├── ai/
│   ├── models/
│   ├── human/
│   ├── vehicle/
│   ├── anpr/
│   ├── face/
│   ├── virtual_fence/
│   ├── suspicious_activity/
│   ├── night_movement/
│   └── pipeline/
│
├── requirements.txt
├── README.md
└── .gitignore