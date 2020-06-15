import flask
import base64
import numpy as np
import cv2
import json


server = flask.Flask(__name__)

query = None
gallery = None


@server.route('/main')
def main():
    return flask.render_template('main_index.html')


# def post_data():
#     global query
#     global gallery
#     print(type(query))
#     print(type(gallery))
#
#     data = flask.request.get_data()
#     data_str = data.decode('utf-8')
#     data_dict = json.loads(data_str)
#
#     if 'img1' in data_dict:
#         query_str = data_dict['img1']
#         query_b64 = query_str.encode('utf-8')
#         query = base64_decode(query_b64)
#         return json.dumps({'msg':'got img1'})
#
#     elif 'img2' in data_dict:
#         gallery = data_dict['img2']
#
#     elif 'msg' in data_dict:
#         if data_dict['msg']=='请求结果':
#             return flask.jsonify({'img1': 'img1', 'img2': 'img2'}), 201
#         else:
#             return 'something wrong'
#     else:
#         return 'something wrong'
#
#     if query:
#     post_img1()


@server.route('/getimg1', methods=['POST'])
def post_img1():
    global query
    global gallery

    data = flask.request.get_data()
    data_str = data.decode('utf-8')
    data_dict = json.loads(data_str)

    query_str = data_dict['img1']
    query_b64 = query_str.encode('utf-8')
    query = base64_decode(query_b64)

    return json.dumps({'msg': 'got img1'})


@server.route('/getimg2', methods=['POST'])
def post_img2():
    global query
    global gallery

    data = flask.request.get_data()
    data_str = data.decode('utf-8')
    data_dict = json.loads(data_str)

    gallery_str = data_dict['img2']
    gallery_b64 = gallery_str.encode('utf-8')
    gallery = base64_decode(gallery_b64)

    return json.dumps({'msg': 'got img2'})


@server.route('/getmsg', methods=['POST'])
def post_msg():
    global query
    global gallery

    data = flask.request.get_data()
    data_str = data.decode('utf-8')
    data_dict = json.loads(data_str)

    msg_str = data_dict['msg']
    msg_b64 = msg_str.encode('utf-8')
    msg = base64_decode(msg_b64)

    if msg == '请求结果':
        # TODO 算法部分
        return flask.jsonify({'img1': 'img1', 'img2': 'img2'}), 201
    else:
        return json.dumps({'msg': 'something wrong'})


def base64_encode(img_bgr):
    img_bgr_str = cv2.imencode('.jpg', img_bgr)[1].tostring()
    img_bgr_b64 = base64.b64encode(img_bgr_str)

    return img_bgr_b64


def base64_decode(img_b64):
    # base64解码
    img_str = base64.b64decode(img_b64)
    img = np.fromstring(img_str, np.uint8)
    img_bgr = cv2.imdecode(img, cv2.IMREAD_COLOR)
    print(img_bgr.shape)

    return img_bgr


if __name__ == '__main__':
    host = '127.0.0.1'
    port = 8080

    debug = True
    server.run(host=host, port=port, debug=debug)