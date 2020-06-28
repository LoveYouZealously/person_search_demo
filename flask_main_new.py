import flask
from werkzeug.datastructures import FileStorage
from flask_restful import Api,Resource,reqparse,request,output_json
import base64
import numpy as np
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import shutil
import json
import time
import sys
import random
import torch
from flask_cors import CORS

import my_create_query_forSocket
import my_search_forSocket_multi
import my_process_image


##########  query model init  &  search model init  ##########
with torch.no_grad():
    dataloader_query, device_query, model_query, classes_query, colors_query = my_create_query_forSocket.query_init()
    dataloader_search, model_search, reidModel, device_search, classes_search, colors_search, weights = my_search_forSocket_multi.search_init()



##########  flask server init  ##########
server = flask.Flask(__name__)

"""
def after_request(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
server.after_request(after_request)
"""

"""
#@server.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,session_id')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS,HEAD')
    # 这里不能使用add方法，否则会出现 The 'Access-Control-Allow-Origin' header contains multiple values 的问题
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

server.after_request(after_request)
"""
#CORS(server,supports_credentials=True)

##########  query init & gallery init  ##########
query = None
gallery = None


@server.route('/main')
def main():
    return flask.render_template('main_index.html')


@server.route('/getimg1', methods=['POST','GET'])
def post_img1():
    print("running func: post_img1")

    global query
    global gallery

    """
    # get_data
    data = flask.request.get_data()
    print('data:', data[:200])
    data_str = data.decode('utf-8')
    print('data_str:', data_str[:200])
    data_dict = json.loads(data_str)
    print(data_dict.keys())

    query_str = data_dict['img1']
    query_str = query_str.replace("data:image/jpeg;base64,", "", 1)
    print(query_str[:200])
    query_b64 = query_str.encode('utf-8')
    query = base64_decode(query_b64)
    """
    print('****img1****')
    req = reqparse.RequestParser()
    req.add_argument('image',required=True,type=FileStorage,location='files')
    args = req.parse_args()
    file_input = args['image']
    fname = file_input.filename.split('.')[0]
    file_input = file_input.read()
    img_input = cv2.imdecode(np.frombuffer(file_input,np.uint8),cv2.IMREAD_COLOR)
    query = img_input
    print('****img1*****')
    
    return json.dumps({"msg": "got img1"})


@server.route('/getimg2', methods=['POST','GET'])
def post_img2():
    print("running func: post_img2")

    global query
    global gallery
    
    req = reqparse.RequestParser()
    req.add_argument('image',required=True,type=FileStorage,location='files')
    args = req.parse_args()
    file_input = args['image']
    fname = file_input.filename.split('.')[0]
    file_input = file_input.read()
    img_input = cv2.imdecode(np.frombuffer(file_input,np.uint8),cv2.IMREAD_COLOR)
    gallery = img_input
    """
    data = flask.request.get_data()
    print('data:', data[:200])
    data_str = data.decode('utf-8')
    print('data_str:', data_str[:200])
    data_dict = json.loads(data_str)
    print(data_dict.keys())

    gallery_str = data_dict['img2']
    gallery_str = gallery_str.replace("data:image/jpeg;base64,", "", 1)
    gallery_b64 = gallery_str.encode('utf-8')
    gallery = base64_decode(gallery_b64)
    """
    return json.dumps({"msg": "got img2"})


@server.route('/getmsg', methods=['POST','GET'])
def post_msg():
    print("running func: msg")

    global query
    global gallery

    if query is None or gallery is None:
        query = None
        gallery = None
        return json.dumps({"msg": "something wrong"})
    print('***',query.shape)
    print('***',gallery.shape)    
    data = flask.request.get_data()
    data_str = data.decode('utf-8')
    data_dict = json.loads(data_str)

    msg_str = data_dict["msg"]
    
    # msg_b64 = msg_str.encode('utf-8')
    # msg = base64_decode(msg_b64)
    
    msg_str = '1'
    try:
        if msg_str == '1':
            ########## create query ##########
            img_res, img = my_process_image.process_img(query)
            dataloader_item = ('query.jpg', img_res, img, None)
            with torch.no_grad():
                query_withBox = my_create_query_forSocket.query_detect(dataloader_item, device_query, model_query, classes_query, colors_query)

            ########## search ##########
            img_res, img = my_process_image.process_img(gallery)
            dataloader_item = ('gallery.jpg', img_res, img, None)
            with torch.no_grad():
                search_begin = time.time()
                gallery_withBox = my_search_forSocket_multi.search_detect(dataloader_item, model_search, reidModel, device_search, classes_search, colors_search, weights)

            query_withBox_b64 = base64_encode(query_withBox)
            query_withBox_str = query_withBox_b64.decode('utf-8')
            gallery_withBox_b64 = base64_encode(gallery_withBox)
            gallery_withBox_str = gallery_withBox_b64.decode('utf-8')

            response = flask.jsonify({"img1": query_withBox_str, "img2": gallery_withBox_str}), 201
        else:
            response = json.dumps({"msg": "something wrong"})
    except Exception as e:
        print(e)
        response = json.dumps({"msg": "something wrong"})
    finally:
        query = None
        gallery = None
        my_create_query_forSocket.query_index = 0
        shutil.rmtree('query')
        os.mkdir('query')

        return response


def base64_encode(img_bgr):
    img_bgr_str = cv2.imencode('.jpg', img_bgr)[1].tostring()
    img_bgr_b64 = base64.b64encode(img_bgr_str)

    return img_bgr_b64


def base64_decode(img_b64):
    
    img_str = base64.b64decode(img_b64)
    print(len(img_str))
    img = np.fromstring(img_str, np.uint8)
    print(img.shape)
    img_bgr = cv2.imdecode(img, cv2.IMREAD_COLOR)
    print(img_bgr.shape)

    return img_bgr


if __name__ == '__main__':
    host = '10.252.97.39'
    port = 9898
    # host = '127.0.0.1'
    # port = 8080

    debug = True
    server.run(host=host, port=port, debug=debug)
