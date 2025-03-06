from pathlib import Path

from ultralytics import YOLO

model_path = Path(__file__).parent / "trained_models" / "yolov11m.onnx"
model = YOLO(str(model_path), task="detect")

inference_dataset_location = (
    Path(__file__).parent / "datasets" / "ball_inference" / "test" / "images"
)
if not inference_dataset_location.exists():
    print("Please run download_datasets.py first to download the dataset.")
    exit()

results_dir = Path(__file__).parent / "results" / model_path.name
results_dir.mkdir(parents=True, exist_ok=True)
results = model(inference_dataset_location, save=True, save_dir=results_dir)
predict_path = None
for r in results:
    path = Path(__file__).parent / r.save_dir / Path(r.path).name
    path.replace(results_dir / Path(r.path).name)
    predict_path = Path(r.path)

predict_path.rmdir()
