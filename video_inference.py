from pathlib import Path

from ultralytics import YOLO
import cv2

model_path = Path(__file__).parent / "trained_models" / "yolov11m.onnx"
model = YOLO(str(model_path), task="detect")

video_path = Path(__file__).parent / "video.mp4"
if not video_path.exists():
    print("Video file not found.")
    exit()

cap = cv2.VideoCapture(str(video_path))
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if (frame_count % 10) == 0:
        results = model(frame, conf=0.25, classes=0)
        if results and results[0].boxes:
            # Find the detection with the highest confidence
            best_box = max(results[0].boxes, key=lambda box: box.conf)
            # Create a new Results object with only the best box
            best_result = results[0].new()
            best_result.boxes = [best_box]
            annotated_frame = best_result.plot()
            cv2.imshow("YOLOv11 Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    frame_count += 1
    if frame_count > 500:
        break

cap.release()
cv2.destroyAllWindows()
