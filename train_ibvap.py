import os
from ultralytics import YOLO

def train_ibvap_model():
    """
    Script to train a custom YOLOv11 model for the Intelligent Border Video Analytics Platform (IBVAP).
    Targets: Humans and Vehicles in border environments.
    """
    print("=== Initializing IBVAP Custom Object Detection Training ===")
    
    # 1. Load a baseline model architecture 
    # 'yolo11n.pt' provides a lightweight, highly efficient architecture ideal for edge/remote deployment.
    model = YOLO("yolo11n.pt")
    
    # 2. Configure training parameters optimized for surveillance
    # - epochs: 50 (Adjust higher up to 100-300 for final production convergence)
    # - imgsz: 640 (Standard surveillance resolution balance between accuracy and FPS)
    # - device: 0 (Uses the first NVIDIA GPU; change to 'cpu' if no GPU is present)
    training_args = {
        "data": "data.yaml",      # Path to dataset config file
        "epochs": 50,             # Number of training epochs
        "imgsz": 640,             # Input image size
        "batch": 16,              # Batch size (adjust based on GPU VRAM)
        "device": 0,              # GPU device ID (0) or 'cpu'
        "workers": 4,             # Number of data loading worker threads
        "augment": True,          # Enable automatic data augmentations (flips, mosaic, blur)
        "project": "IBVAP_Runs",   # Save directory name
        "name": "human_vehicle_detector"
    }
    
    print("\nStarting training loop with parameters:")
    for key, val in training_args.items():
        print(f"  - {key}: {val}")
        
    try:
        # Launch custom training pipeline
        results = model.train(**training_args)
        print("\n=== Training Completed Successfully! ===")
        print("Best model weights saved to: IBVAP_Runs/human_vehicle_detector/weights/best.pt")
    except Exception as e:
        print(f"\n[Error during training]: {e}")
        print("Ensure CUDA is configured if utilizing device=0, and dataset directories contain images.")

if __name__ == "__main__":
    train_ibvap_model()