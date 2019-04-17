from config.processing import *
import dlib

face_detector_hog = dlib.get_frontal_face_detector()
face_detector_cnn = dlib.cnn_face_detection_model_v1('./models/mmod_human_face_detector.dat')
shape_predictor_68 = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
shape_predictor_5 = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
face_recognition = dlib.face_recognition_model_v1('./models/dlib_face_recognition_resnet_model_v1.dat')


def face_detection(image, model='HOG', iter=1):
    """
    :param image: image in numpy array
    :param model: face detection model HOG or CNN
    :param iter: face find iteration
    :return: array of faces [top, left, bottom, right]
    """
    face_detector = face_detector_hog
    faces = []
    if model == 'CNN':
        face_detector = face_detector_cnn
        face_location = face_detector(image, iter)

        for face in face_location:
            a, b, c, d = face.rect.top(), face.rect.left(), face.rect.bottom(), face.rect.right()
            faces.append([a, c, b, d])
    else:
        face_location = face_detector(image, iter)
        for face in face_location:
            a, b, c, d = face.left(), face.right(), face.top(), face.bottom()
            faces.append([a, c, b, d])

    return faces


def landmark_detection(image, face, model="68_point"):
    """
    :param image: image in numpy array
    :param model: landmark detection model, 68_pont or 5_point
    :return: face landmarks array
    """
    face = dlib.rectangle(*face)

    shape = shape_predictor_68(image, face)

    landmarks = []

    for shp in shape.parts():
        landmarks.append(shp)

    return landmarks


def face_encoder(image, face, iter=3):
    """
    :param image:
    :param face:
    :param iter:
    :return: 128 dimension array
    """
    face = dlib.rectangle(*face)
    shape = shape_predictor_68(image, face)
    face_code = face_recognition.compute_face_descriptor(image, shape, iter)

    return np.array(face_code)

def compare_faces(face1_code, face2_code, dist = 0.6):
    """
    :param face1_code: fisrt face code
    :param face2_code: second face code
    :param dist: maximal dist between same faces
    :return: True if face is same False otherwise
    """
    return np.linalg.norm(face1_code-face2_code, axis=0) <= dist

