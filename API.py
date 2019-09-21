from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from face_detection.face_detector import *
from config.processing import *
from io import BytesIO
from PIL import Image
import numpy as np

app = Flask(__name__)
api = Api(app)

class Compare_faces(Resource):
    def get(self):
        data = request.files.to_dict()
        try:
            file1 = data['image_1'].read()
            file2 = data['image_2'].read()

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
        except:
            return jsonify({"dist" : None, "verdict" : None})

class Get_faces(Resource):
    def get(self):
        data = request.files.to_dict()
        try:
            File = data['image'].read()

            with Image.open(BytesIO(File)) as img:
                img = np.array(img.convert('RGB'))

            faces = face_detection(img, model="HOG", iter=2)

            print(type(faces), faces)
        
            return jsonify({"faces" : faces})
        except:
            return jsonify({"faces" : None})

api.add_resource(Compare_faces, '/compare')
api.add_resource(Get_faces, '/get_faces')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)