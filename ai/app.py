from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np

from ai.yolo import process_img


app = Flask(__name__)
CORS(app)

@app.route('/')
def a():
    return 'hello'

@app.route('/image/classification', methods=['POST'])
def classify_img():
    img_file = request.files['image']

    np_img = np.frombuffer(img_file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    result = process_img(img)

    return result

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", port=5002)