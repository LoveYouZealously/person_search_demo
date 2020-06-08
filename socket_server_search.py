import os
import socket
import pickle
import time
import sys
import numpy as np
import random
import cv2
import torch

import my_search_forSocket
import my_process_image


########## search model init ##########
with torch.no_grad():
    dataloader, model, reidModel, device, classes, colors, weights = my_search_forSocket.search_init()

########## server starting ##########
server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host = '10.251.133.5'
port = 9351
server.bind((host, port))

server.listen(1)
print('服务已启动')


my_search_forSocket.query_time_last = time.time()
my_search_forSocket.query_time_now = my_search_forSocket.query_time_last
my_search_forSocket.query_feats = []
my_search_forSocket.query_pids  = []

count = 0
while True:
    conn, addr = server.accept()
    while True:
        try:
            data = conn.recv(2**16)
            if data:
                if b'ClosedByClient' in data:
                    print('客户端正常关闭了链接')
                if b'EndOfFileFFF' in data:
                    print('???')
                    pass
                else:
                    parti = 0
                    datas = data
                    begin_gotPart = time.time()
                    while data:
                        data = conn.recv(2**16)
                        datas += data 
                        parti += 1
                        print('got part:', parti, end='\r')
                        if b'EndOfFileFFF' in data:
                            print('got part:', parti)
                            print('got EndOfFileFFF')
                            break

                print('gotPart_time:', time.time()-begin_gotPart)
                package_dict = pickle.loads(datas)
                print('dict got')

                ########## detect ##########
                img = package_dict['img_np']
                processImg_begin = time.time()
                img_res, img = my_process_image.process_img(img)
                print('processImg_time:', time.time()-processImg_begin)
                dataloader_item = ('webcam_{}.jpg'.format(count), img_res, img, None)
                with torch.no_grad():
                    search_begin = time.time()
                    img_withBox = my_search_forSocket.search_detect(dataloader_item, model, reidModel, device, classes, colors, weights)
                    print('search_time:', time.time()-search_begin)
                print(img_withBox.shape)
                response = pickle.dumps({'img_np':img_withBox})

                send_begin = time.time()
                conn.send(response+b'EndOfFileFFF')
                print('send_time:', time.time()-send_begin)
                print('total_time:', time.time()-begin_gotPart)
                print('########## response send ##########')
        except ConnectionResetError as e:
            print('客户端强制关闭了链接')
            break




