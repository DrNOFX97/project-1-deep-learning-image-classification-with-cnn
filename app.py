import os
import random
import requests
from flask import Flask, request, render_template_string, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model = MobileNetV2(weights='imagenet')

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
            background-image: url('{{ bg_image }}');
            background-size: cover;
            padding: 100px;
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
            background-image: url('{{ bg_image }}');
            background-size: cover;
            padding: 100px;
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
            <h1 class="display-4">This seems to be...</h1>
        </div>
        {% for idx in range(filenames|length) %}
        <div class="result-container">
            <h2>Uploaded Image: {{ filenames[idx] }}</h2>
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

bg_image_urls = []


def download_random_images(num_images):
    global bg_image_urls
    for _ in range(num_images):
        response = requests.get('https://picsum.photos/1200/800')
        if response.status_code == 200:
            filename = f"{random.randint(10000, 99999)}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            bg_image_urls.append(filepath)


def choose_random_bg_image(bg_image_urls):
    return random.choice(bg_image_urls)


@app.route('/')
def index():
    return render_template_string(home_template, bg_image=choose_random_bg_image(bg_image_urls))


@app.route('/loading')
def loading():
    return render_template('loading.html')


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

            image = load_img(filepath, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)

            preds = model.predict(image)
            decoded_preds = decode_predictions(preds, top=5)[0]

            predictions = [{'class_id': i + 1, 'class_name': pred[1], 'probability': round(pred[2] * 100, 2)} for
                           i, pred in enumerate(decoded_preds)]
            predictions_list.append(predictions)

    return render_template_string(results_template, filenames=filenames,
                                   predictions_list=predictions_list,
                                   bg_image=choose_random_bg_image(bg_image_urls))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    # Download 50 random images for background
    download_random_images(50)
    app.run(debug=True)