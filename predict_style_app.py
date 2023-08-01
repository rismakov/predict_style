from flask import Flask, request, jsonify

from tensorflow.keras.models import load_model

import cv2  # for image processing
import numpy as np

from PIL import Image

app = Flask(__name__)

PATTERN_MODEL = load_model('model.hdf5')

# ADD TO FLUTTER APP:
#       // Send the image to the Python server
#       final url = 'YOUR_PYTHON_SERVER_URL'; // Replace with your Python server URL
#       final response = await http.post(Uri.parse(url), body: {
#         'image': _image!.path,
#       });

def resize_and_flatten_image(image_path):
    """
    Convert image to an array of size 100x1.

    The array represents an OpenCV grayscale version of the original image.
    The image will get cropped along the biggest red contour (4 line polygon)
    tagged on the original image (if any).
    """
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_100x100 = cv2.resize(image_gray, (100, 100))

    return image_100x100.flatten()

# Function to process the image and get the predicted label
def process_images(image_files):
    processed_images = []
    for image_file in image_files:
        processed_images.append(resize_and_flatten_image(image_file))

    X = np.array(images_processed) / 255
    img_rows, img_cols = 100, 100
    X = X.reshape(X.shape[0], img_rows, img_cols, 1)

    predictions = pattern_model.predict(X)
    binary_labels = []
    for prediction in predictions:
        label = np.argmax(prediction)
        if label == 8:
            binary_labels.append('Basic')
        else:
            binary_labels.append('Statement')

    return binary_labels

# Route to handle the image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the request contains files
        if 'images[]' not in request.files:
            return jsonify({'error': 'No images found in the request'}), 400

        # Get the list of files from the request
        image_files = request.files.getlist('images[]')

        # Process each image and get the predicted labels
        predicted_labels = process_images(image_files)

        # Return the list of predicted labels as a response
        return jsonify({'predicted_labels': predicted_labels}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
