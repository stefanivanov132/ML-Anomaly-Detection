# ML-Anomaly-Detection
Machine Learning And Data Mining - Final Project

# CNN Model Project

This repository contains a trained Convolutional Neural Network (CNN) model saved as a pickle file inside the `model_artifact` folder. The model is ready for use and can be loaded on any machine using the provided instructions.

## Project Structure
```
/project_directory/
│── model_artifact/
│   ├── cnn_model.pkl
│── requirements.txt
│── your_script.py
│── user_guide.txt
│── README.md
```

## Getting Started

### Prerequisites
Make sure you have **Python 3.8 or later** installed.

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
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

## Notes
- Keep the `cnn_model.pkl` file inside the `model_artifact` folder.
- If you encounter issues with missing dependencies, manually install them using `pip install package_name`.

## License
This project is licensed under the MIT License.

## Author
[Your Name](https://github.com/yourusername)

---

For a complete guide, see `user_guide.txt` in the project root.

Happy coding! ✨

