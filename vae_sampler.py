#!/usr/bin/env python3.9

import numpy as np
import cv2
import math

track = cv2.imread('track.png')

width, height,_ = track.shape

number_to_generate = 12
image_res = (640,480)

def random_image():
    alpha = np.random.uniform(0,2*math.pi)
    x = np.random.uniform(0,width-image_res[0])
    y = np.random.uniform(0,height-image_res[1])
    transform = np.array([[math.cos(alpha), -math.sin(alpha),x],[math.sin(alpha), math.cos(alpha),y]])
    transformed = cv2.warpAffine(track,transform, (image_res[0], image_res[1]))
    if np.where(transformed > 200)[0].shape[0] > 0:
        return np.array(transformed)
    else:
        return None

cnt = 0
while True:
    image = random_image()
    if image is not None:
        image = cv2.resize(image, (160,120))
        cv2.imwrite(f'./vae_data/{cnt:05d}.jpg', image)
        cnt+=1
    if cnt >= number_to_generate:
        break
    
