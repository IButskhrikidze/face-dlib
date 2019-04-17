import dlib
import numpy as np
from PIL import Image, ImageDraw, ImageFile


def read_image(path, mode='RGB'):
    """
    :param path: image location
    :param mode: image mode RGB or GRAY
    :return: numpy array
    """
    try:
        if mode == 'RGB':
            return dlib.load_rgb_image(path)
        return dlib.load_grayscale_image(path)
    except:
        print("path is invalid")


def show_image(img):
    """
    :param img: image in numpy array
    :return: show image in display
    """
    img = Image.fromarray(img)
    img.show()


def rectangle_image(img, location=[0, 0, 100, 100], color=(255, 0, 0)):
    """
    :param img: image in numpy array
    :param location: tuple of rectangle location (top,left,bottom,right)
    :param color: tuple of color RGB mode. default color is red (255, 0, 0)
    :return: return image in numpy array with a rectangle
    """
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.rectangle(location, outline=color)
    return np.array(img)


def ellipse_image(img, location=[0, 0, 100, 100], color=(255, 0, 0)):
    """
    :param img: image in numpy array
    :param location: tuple of rectangle location (top,left,bottom,right)
    :param color: tuple of color RGB mode. default color is red (255, 0, 0)
    :return: return image in numpy array with a rectangle
    """
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.ellipse(location, outline=color)
    return np.array(img)


def save_image(img, path):
    """
    :param img: image in numpy array
    :param path: location where want save image
    :return: saved image
    """
    img = Image.fromarray(img)
    ImageFile.MAXBLOCK = img.size[0] * img.size[1]
    img.save(path, "JPEG", quality=100, optimize=True, progressive=True)


def cut_face(img, face):
    """
    :param img: takes image in numpy array
    :param face: face location array [top, left, bottom, right]
    :return: face in numpy array
    """
    a, b, c, d = tuple(face)
    return img[b:d, a:c]


def mark_landmarks(image, landmarks, color=255):
    """
    :param image: image in numpy array
    :param landmarks: landmark points
    :param color: landmark marker color
    :return: landmark markered image
    """
    for landmark in landmarks:
        image[landmark.y][landmark.x] = color
        image[landmark.y + 1][landmark.x] = color
        image[landmark.y - 1][landmark.x] = color
        image[landmark.y][landmark.x + 1] = color
        image[landmark.y][landmark.x - 1] = color

    return image
