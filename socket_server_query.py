import os
import socket
import pickle
import time
import sys
import numpy as np
import random
import cv2
import torch

import my_create_query_forSocket
import my_process_image


########## search model init ##########
with torch.no_grad():
    dataloader, device, model, classes = my_create_query_forSocket.query_init()

########## server starting ##########
server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host = '10.251.133.5'
port = 9350
server.bind((host, port))

server.listen(1)
print('服务已启动')


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
                    while data:
                        data = conn.recv(2**16)
                        datas += data 
                        parti += 1
                        print('get part:', parti, end='\r')
                        if b'EndOfFileFFF' in data:
                            print('get part:', parti)
                            print('got EndOfFileFFF')
                            break

                package_dict = pickle.loads(datas)
                print('dict got')

                ########## detect ##########
                img = package_dict['img_np']
                img_res, img = my_process_image.process_img(img)
                dataloader_item = ('webcam_{}.jpg'.format(count), img_res, img, None)
                with torch.no_grad():
                    crop_img = my_create_query_forSocket.query_detect(dataloader_item, device, model, classes)
                # if crop_img is not None:
                    # print(crop_img.shape)
                response = pickle.dumps({'img_np':crop_img})

                conn.send(response+b'EndOfFileFFF')
                print('response send')
        except ConnectionResetError as e:
            print('客户端强制关闭了链接')
            break




