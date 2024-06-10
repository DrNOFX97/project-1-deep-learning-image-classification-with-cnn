import os
import ssl
from flask import Flask, request, render_template_string, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model = MobileNetV2(weights='imagenet')

home_template = '''
<!doctype html>
<html>
<head>
    <title>Flask Project - Image prediciton</title>
</head>
<body>
    <h1>WHAT IS IT??</h1>
    <h2>Image prediction webapp</h2>
    <h3>Pick an image</h3>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="files[]" multiple>
        <input type="submit" value="Upload">
    </form>
    <footer>
        <p>IronHackers: Nour Kashto / Fernando Nuno Vieira</p>
    </footer>
</body>
</html>
'''

results_template = '''
<!doctype html>
<html>
<head>
    <title>AI Prediction Results</title>
    <style>
        table {
            width: 50%;
            border-collapse: collapse;
        }
        table, th, td {
            border: 1px solid black;
        }
        th, td {
            padding: 8px;
            text-align: left;
        }
        .image-result {
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <h1>Prediction Results</h1>
    {% for idx in range(filenames|length) %}
    <div class="image-result">
        <h2>Uploaded Image: {{ filenames[idx] }}</h2>
        <img src="{{ url_for('uploaded_file', filename=filenames[idx]) }}" alt="Uploaded Image" style="width:300px;">
        <h3>Predictions</h3>
        <table>
            <tr>
                <th>Class</th>
                <th>Class Name</th>
                <th>Probability</th>
            </tr>
            {% for prediction in predictions_list[idx] %}
            <tr>
                <td>{{ prediction['class_id'] }}</td>
                <td>{{ prediction['class_name'] }}</td>
                <td>{{ prediction['probability'] }}%</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endfor %}
    <a href="/">Upload more images</a>
    <footer>
        <p>Disclaimer: This is a deep learning focused app, layout is not a priority.</p>
    </footer>
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

            image = load_img(filepath, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)

            preds = model.predict(image)
            decoded_preds = decode_predictions(preds, top=5)[0]

            predictions = [{'class_id': i + 1, 'class_name': pred[1], 'probability': round(pred[2] * 100, 2)} for
                           i, pred in enumerate(decoded_preds)]
            predictions_list.append(predictions)

    return render_template_string(results_template, filenames=filenames, predictions_list=predictions_list)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=True)
