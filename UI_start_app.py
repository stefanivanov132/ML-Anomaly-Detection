import os
import pickle
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from PIL import Image

class TumorCNN(nn.Module):
    def __init__(self):
        super(TumorCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def main():
    model_path = os.path.join("model_artifact", "cnn_model.pkl")

    with open(model_path, "rb") as f:
        cnn_model = pickle.load(f)
        
    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    st.title("Tumor Detection with CNN")
    st.write("Upload a histopathology image, and the model will predict if it contains a tumor.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = cnn_model(image_tensor)
            _, predicted_class = torch.max(output, 1)

        result = "Tumor Detected" if predicted_class.item() == 1 else "Normal Tissue"
        st.write(f"### Prediction: {result}")

if __name__ == '__main__':
    main()