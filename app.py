from flask import Flask, render_template, request, send_file
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import requests
import os
import io
app = Flask(__name__)
# Load the pretrained ResNet-18 model
model = models.resnet18(pretrained=True)
model.eval()
# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Function to download class names file
def download_classes_file(url, destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            f.write(response.content)
    else:
        print("Failed to download class names file.")
# Download class names file
classes_url = "https://raw.githubusercontent.com/joannzhang00/ImageNet-dataset-classes-labels/main/imagenet_classes.txt"
classes_file = "imagenet_classes.txt"
download_classes_file(classes_url, classes_file)
# Load class names
with open(classes_file, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
# Function to predict image class
def predict_image(image_path):
    try:
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = None
    image_data = None
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return render_template('index.html', error='No file uploaded.')
        if file.filename == '':
            return render_template('index.html', error='No file selected.')
        if file:
            file_path = 'static/uploaded_image.jpg'
            file.save(file_path)
            prediction = predict_image(file_path)
            image_data = io.BytesIO()
            Image.open(file_path).save(image_data, "JPEG")
            image_data.seek(0)
    return render_template('index.html', prediction=prediction, image_data=image_data)
if __name__ == '__main__':
    app.run(debug=True)