import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

root = Path(__file__).parent

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("roboflow-universe-projects").project(
    "basketball-players-fy4c2"
)
version = project.version(16)
dataset_location = root / "datasets" / "ball_fine_tune"
if not dataset_location.exists():
    dataset_location.mkdir(parents=True, exist_ok=True)
dataset = version.download("yolov11", str(dataset_location), overwrite=True)

project = rf.workspace("james-skelton").project("ballhandler-basketball")
version = project.version(11)
inferece_dataset_location = root / "datasets" / "ball_inference"
if not inferece_dataset_location.exists():
    inferece_dataset_location.mkdir(parents=True, exist_ok=True)
dataset = version.download(
    "yolov11", str(inferece_dataset_location), overwrite=True
)
