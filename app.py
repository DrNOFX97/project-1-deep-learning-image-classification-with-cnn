import os
import ssl
from flask import Flask, request, render_template_string, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load the MobileNetV2 model pre-trained on ImageNet
model = MobileNetV2(weights='imagenet')

# Define the classes we care about (CIFAR-10)
cifar10_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# A mapping of CIFAR-10 class names to the closest ImageNet class indices
cifar10_to_imagenet_indices = {
    'airplane': 404,     # Index for 'airliner'
    'automobile': 817,   # Index for 'sports_car'
    'bird': 16,          # Index for 'partridge'
    'cat': 281,          # Index for 'tabby_cat'
    'deer': 349,         # Index for 'ox'
    'dog': 208,          # Index for 'Golden_retriever'
    'frog': 31,          # Index for 'tree_frog'
    'horse': 340,        # Index for 'sorrel'
    'ship': 510,         # Index for 'liner'
    'truck': 569         # Index for 'trailer_truck'
}

# Template for the homepage
home_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Prediction App</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body {
            background-color: #f8f9fa;
            padding-top: 50px;
        }
        .container {
            max-width: 600px;
        }
        .jumbotron {
            padding: 100px;
            background-color: #007bff;
            color: white;
            text-align: center;
        }
        .form-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.1);
            margin-top: 50px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="jumbotron">
            <h1 class="display-4">WHAT IS IT?</h1>
            <p class="lead">Upload an image to get predictions</p>
        </div>
        <div class="form-container">
            <form action="/upload" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <input type="file" class="form-control-file" name="files[]" multiple>
                </div>
                <button type="submit" class="btn btn-primary">Upload</button>
            </form>
        </div>
    </div>
</body>
</html>
'''

# Template for the results page
results_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Results</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        body {
            background-color: #f8f9fa;
            padding-top: 50px;
        }
        .container {
            max-width: 800px;
        }
        .jumbotron {
            padding: 100px;
            background-color: #28a745;
            color: white;
            text-align: center;
        }
        .result-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.1);
            margin-top: 50px;
        }
        img {
            max-width: 100%;
            height: auto;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="jumbotron">
            <h1 class="display-4">WHAT IS IT?</h1>
        </div>
        {% for idx in range(filenames|length) %}
        <div class="result-container">
            <h2>This seems to be a {{ predictions_list[idx][0]['class_name'] }}</h2>
            <img src="{{ url_for('uploaded_file', filename=filenames[idx]) }}" alt="Uploaded Image">
            <h3>Predictions</h3>
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th scope="col">Class</th>
                        <th scope="col">Class Name</th>
                        <th scope="col">Probability</th>
                    </tr>
                </thead>
                <tbody>
                    {% for prediction in predictions_list[idx] %}
                    <tr>
                        <td>{{ prediction['class_id'] }}</td>
                        <td>{{ prediction['class_name'] }}</td>
                        <td>{{ prediction['probability'] }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endfor %}
        <div class="form-container">
            <a href="/" class="btn btn-primary">Upload more images</a>
        </div>
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(home_template)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return 'No files part in the request', 400

    files = request.files.getlist('files[]')

    if not files:
        return 'No files uploaded', 400

    filenames = []
    predictions_list = []
    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            filenames.append(filename)

            # Preprocess the image
            image = load_img(filepath, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)

            # Predict the image class
            preds = model.predict(image)
            
            # Extract the probabilities for CIFAR-10 classes
            filtered_preds = []
            for i, class_name in enumerate(cifar10_classes):
                imagenet_index = cifar10_to_imagenet_indices[class_name]
                prob = preds[0][imagenet_index]
                filtered_preds.append({
                    'class_id': i + 1,
                    'class_name': class_name,
                    'probability': round(prob * 100, 2)
                })

            # Sort the filtered predictions by probability
            filtered_preds = sorted(filtered_preds, key=lambda x: x['probability'], reverse=True)
            predictions_list.append(filtered_preds)

    return render_template_string(results_template, filenames=filenames, predictions_list=predictions_list)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
