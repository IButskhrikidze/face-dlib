import requests

#request structure for get_faces
def send_req1(path):
    r = requests.get(
        "http://192.168.0.64:5000/get_faces",

        files = {
            'image' : open(path, 'rb')
        },

        data = {
            'token' : 'ibutskhrikidze',
            'iter' : 1,
            'model' : 'HOG'
        }
    )

    print(r.json())

#request structure for face_compare

def send_req2(path1, path2):
    r = requests.get(
        "http://192.168.0.64:5000/compare",

        files = {
            'image_1' : open(path1, 'rb'),
            'image_2' : open(path2, 'rb'),
        },

        data = {
            'token' : 'ibutskhrikidze',
            'iter' : 1,
            'model' : 'HOG',
            'threshold' : 0.6,
        }
    )

    print(r.json())

send_req2('./image/img1.jpg', './image/img2.jpg')