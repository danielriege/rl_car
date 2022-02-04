#!/usr/bin/env python3.9

import numpy as np
import cv2
import math

class VAE_Sampler:
    def __init__(self, track_path, image_res=(640,480)):
        self.__track = cv2.imread(track_path)

        self.__width, self.__height,_ = self.__track.shape

        self.__image_res = image_res#(640,480)

    def random_image(self):
        alpha = np.random.uniform(0,2*math.pi)
        x = np.random.uniform(0,self.__width-self.__image_res[0])
        y = np.random.uniform(0,self.__height-self.__image_res[1])
        transform = np.array([[math.cos(alpha), -math.sin(alpha),x],[math.sin(alpha), math.cos(alpha),y]])
        transformed = cv2.warpAffine(self.__track,transform, self.__image_res)
        if np.where(transformed > 200)[0].shape[0] > 0:
            return np.array(transformed)
        else:
            return None

if __name__ == "__main__":
    sampler = VAE_Sampler('./track.png', (640,480))
    cnt = 0
    while True:
        image = sampler.random_image()
        if image is not None:
            image = cv2.resize(image, (160,120))
            cv2.imwrite(f'./{cnt:05d}.jpg', image)
            print(np.max(image))
            cnt+=1
        if cnt >= 1:
            break
        
