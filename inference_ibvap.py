import cv2
from ultralytics import YOLO

def run_live_inference(video_source=0, weights_path="IBVAP_Runs/human_vehicle_detector/weights/best.pt"):
    """
    Applies the trained IBVAP custom model to a live camera feed or an RTSP stream.
    """
    print(f"Loading custom IBVAP weights from: {weights_path}")
    try:
        # Load custom model. Fallback to base model if custom training hasn't run yet.
        model = YOLO(weights_path)
    except Exception:
        print("Custom weights not found. Falling back to default baseline model for demonstration...")
        model = YOLO("yolo11n.pt")
        
    # Open the video stream (0 for webcam, or pass "rtsp://username:password@ip_address:port/h264")
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    print("Streaming started. Press 'q' to exit the monitoring feed.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from video source.")
            break
            
        # Run real-time tracking inference (persists IDs across frames for tracking requirement)
        results = model.track(source=frame, persist=True, conf=0.4, verbose=False)
        
        # Draw bounding boxes, class labels, and tracking IDs directly on the frame
        annotated_frame = results[0].plot()
        
        # Display the live intelligent border surveillance monitor
        cv2.imshow("IBVAP Live Surveillance - Intelligent Tracking Feed", annotated_frame)
        
        # Break loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Surveillance stream closed.")

if __name__ == "__main__":
    # To run on a sample border video or RTSP stream, change 0 to "path_to_video.mp4"
    run_live_inference(video_source=0)