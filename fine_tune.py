from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

if __name__ == "__main__":
    print("GPU Available: ", torch.cuda.is_available())
    ultralytics.checks()
    dataset_location = Path(__file__).parent / "datasets" / "ball_fine_tune"
    if not dataset_location.exists():
        print("Please run download_datasets.py first to download the dataset.")
        exit()

    model_size = "m"
    model = YOLO(f"yolo11{model_size}.pt")

    train_results = model.train(
        data=f"{dataset_location}/data.yaml", epochs=50, imgsz=640, device="0"
    )

    path = model.export(format="onnx")
    trained_models = Path(__file__).parent / "trained_models"
    trained_models.mkdir(parents=True, exist_ok=True)
    Path(path).replace(trained_models / f"yolov11{model_size}.onnx")
    print(
        f"Model saved to {trained_models / f'yolov11{model_size}_tuned.onnx'}"
    )
