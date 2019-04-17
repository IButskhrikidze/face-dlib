from face_detection.face_detector import *
from config.processing import *

image1 = read_image('./image/img1.jpg')
image2 = read_image('./image/img2.jpg')

faces1 = face_detection(image1)
faces2 = face_detection(image2)

code1 = face_encoder(image1, faces1[0], 5)
code2 = face_encoder(image2, faces2[0], 5)

import numpy as np

print(np.linalg.norm(code1-code2, axis=0))

print(compare_faces(code1, code2))

