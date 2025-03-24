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

## Using the Model
To use the project run the following command:
```bash
streamlit run UI_start_app.py
```

## Making Predictions
Once loaded, you can use the model to make predictions by uploading a histopathology image.
The model would predict if the image contains a tumor.


