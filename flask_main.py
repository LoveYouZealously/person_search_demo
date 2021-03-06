import flask
import base64
import numpy as np
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import time
import sys
import random
import torch

import my_create_query_forSocket
import my_search_forSocket
import my_process_image


##########  query model init  &  search model init  ##########
with torch.no_grad():
    dataloader_query, device_query, model_query, classes_query = my_create_query_forSocket.query_init()
    dataloader_search, model_search, reidModel, device_search, classes_search, colors, weights = my_search_forSocket.search_init()


server = flask.Flask(__name__)

query = None
gallery = None


@server.route('/main')
def main():
    return flask.render_template('main_index.html')


@server.route('/main', methods=['POST'])
def post_data():
    global query
    global gallery
    print(type(query))
    print(type(gallery))

    data = flask.request.get_data()
    if data.decode('utf-8') == '{msg: "请求结果"}':
        #  query_withBox, query_cut = query_algo(query)  TODO 运行query算法
        #  gallery_withBox = search(query, gallery)  TODO 运行search算法，得到结果
        
        if query is None or gallery is None:
            return "something wrong"
        
        ########## create query ##########
        img_res, img = my_process_image.process_img(query)
        dataloader_item = ('query.jpg', img_res, img, None)
        with torch.no_grad():
            crop_img, query_withBox = my_create_query_forSocket.query_detect(dataloader_item, device_query, model_query, classes_query)
        
        ########## search ##########
        img_res, img = my_process_image.process_img(gallery)
        dataloader_item = ('gallery.jpg', img_res, img, None)
        with torch.no_grad():
            search_begin = time.time()
            gallery_withBox = my_search_forSocket.search_detect(dataloader_item, model_search, reidModel, device_search, classes_search, colors, weights)

        query_withBox_b64 = base64_encode(query_withBox)
        query_withBox_str = query_withBox_b64.decode('utf-8')
        gallery_withBox_b64 = base64_encode(gallery_withBox)
        gallery_withBox_str = gallery_withBox_b64.decode('utf-8')

        query = None
        gallery = None

        return flask.jsonify({'img1': query_withBox_str, 'img2': gallery_withBox_str}), 201
    else:
        img_bgr = base64_decode(data)

        # cv2.imshow("img", img_bgr)
        # cv2.waitKey(0)

        if query is None:
            query = img_bgr
            return 'got img1'
        elif gallery is None:
            gallery = img_bgr
            return 'got img2'
        else:
            return "something wrong"  # TODO


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
    # host = '10.251.133.5'
    # port = 9898

    host = '10.252.97.39'
    port = 9898

    debug = True
    server.run(host=host, port=port, debug=debug)
