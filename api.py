from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from flask_jwt import JWT
from PIL import Image
from io import BytesIO
import numpy as np
from face_detection.face_detector import *
from config.processing import *


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def check():
    data = request.files.to_dict()

    file1 = data['file'].read()
    file2 = data['file1'].read()

    with Image.open(BytesIO(file1)) as img:
        img1 = np.array(img.convert('RGB'))

    with Image.open(BytesIO(file2)) as img:
        img2 = np.array(img.convert('RGB'))

    faces1 = face_detection(img1)
    faces2 = face_detection(img2)

    code1 = face_encoder(img1, faces1[0], 5)
    code2 = face_encoder(img2, faces2[0], 5)
    ans = compare_faces(code1, code2, 0.5)
    return jsonify(ans)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)
