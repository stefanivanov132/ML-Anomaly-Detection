# ML-Anomaly-Detection

This repository contains a trained Convolutional Neural Network (CNN) model saved as a pickle file inside the `model_artifact` folder. The model is ready for use and can be loaded on any machine using the provided instructions.

## Project Structure
```
/project_directory/
│── model_artifact/
│   ├── cnn_model.pkl
│── requirements.txt
│── UI_start_app.py
│── README.md
```

## Getting Started

### Prerequisites
Make sure you have **Python 3.8 or later** installed.

### Installation
1. Clone the repository:
```bash
git clone https://github.com/stefanivanov132/ML-Anomaly-Detection.git
cd ML-Anomaly-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Loading the Model
Use the following code snippet in your script to load the trained model:

```python
import os
import pickle

model_path = os.path.join("model_artifact", "cnn_model.pkl")

with open(model_path, "rb") as f:
    cnn_model = pickle.load(f)

print("Model loaded successfully!")
```

## Making Predictions
Once loaded, you can use the model to make predictions:
```python
predictions = cnn_model.predict(X_test)
print("Predictions:", predictions)
```

