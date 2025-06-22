# Detection and Classification of External Bovine Diseases Using Deep Learning

This repository contains the training notebook and dataset used for a Final Year Project (FYP) focused on developing a deep learning model to detect and classify visible bovine diseases. The model is based on the YOLOv11 architecture, chosen for its real-time performance, high precision, and suitability for agricultural applications.

## 📁 Contents
Datasets:
Compressed version of the dataset curated for training. Includes images and YOLO-formatted labels for diseases like Lumpy Skin Disease (LSD), Foot and Mouth Disease (FMD), Ringworm, and Infectious Bovine Keratoconjunctivitis (IBK).
- `Bovine Disease Detection-Combined.v2i.yolov11.zip`
- `Cattle Body Part.v2i.yolov11` 
- `Bovine Body Disease Detection.v2i.yolov11.zip`
- `Bovine Head Disease Detection.v1i.yolov11.zip`

These dataset can also be accessed through Roboflow Universe:
- https://universe.roboflow.com/fyp-coz6v/bovine-disease-detection-combined
- https://universe.roboflow.com/fyp-coz6v/bovine-body-part
- https://universe.roboflow.com/fyp-coz6v/bovine-body-disease-detection
- https://universe.roboflow.com/fyp-coz6v/bovine-head-disease-detection

Jupyter Notebook:
- `bovine_disease_training.ipynb`: Jupyter notebook used for training and evaluating the YOLOv11 model.

This file:
- `README.md`

## 📊 Project Overview

The goal of this project is to classify and localize bovine diseases from visible symptoms using deep learning object detection techniques. The model is trained to recognize diseases in different parts of the animal (e.g., head or body), and a multi-model approach is used to improve performance.

This project includes:
- Custom dataset collection and annotation
- YOLOv11 model training using Ultralytics
- Hyperparameter tuning and model evaluation

## 📦 Dataset

The dataset includes images and bounding box annotations formatted for YOLO. It has been split into:
- Head Disease Dataset (IBK, FMD)
- Body Disease Dataset (LSD, Ringworm)
- Combined Dataset (for baseline model)

Preprocessing and augmentation were applied to improve generalization.

## ⚙️ How to Recreate This Project

### Prerequisites

- Python 3.8+
- Google Colab (or Jupyter Notebook with GPU support)
- Roboflow account (for easy dataset handling)
- `ultralytics` library
- `roboflow` Python package

### Installation

1. Clone this repository or upload the notebook to [Google Colab](https://colab.research.google.com).
2. Install the required libraries:

```bash
pip install ultralytics roboflow supervision
```

3. Download the dataset:
- You can unzip dataset.zip and manually load it in the notebook.
- Or import from Roboflow using the provided API key in the notebook (recommended for repeatability).

### Training Steps
1. Run the notebook bovine_disease_training.ipynb from top to bottom.
2. Set your training config, e.g.:
```Python
!yolo task=detect mode=train model=yolo11s.pt data=your_dataset_path/data.yaml epochs=100 imgsz=640 optimizer=AdamW cos_lr=True patience=10
```
3. Evaluate using mAP, precision and recall
4. Export model:
```Python
model.export(format="onnx")
```
- You can upload your trained weights to Ultralytics Hub or deploy using Roboflow API/SDK.
```Python
project.version(dataset.version).deploy(model_type="yolov11", model_path=f"{HOME}/runs/detect/train/")
```
## 📈 Results
The final model achieved high detection performance (mAP>90%) using a multi-model approach, trained on region-specific datasets (head and body) with augmentation.

## 📜 License
This repository is for academic and non-commercial research purposes.

## 🙋‍♂️ Author
Muhamad Faaris Bin Jamhari
Faculty of Computer Science and Information Technology
Universiti Malaysia Sarawak
