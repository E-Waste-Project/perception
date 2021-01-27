#!/usr/bin/env python
import rospy
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import cv2
from keras.models import load_model
import numpy as np

def load_images(path):
    temp_img,temp_mask=[],[]
    images=glob(os.path.join(path,'*.jpg'))
    #images=images.sort()
    for i in tqdm(sorted(images)):
        i=cv2.imread(i)
        # i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        i=cv2.normalize(i,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
        i=cv2.resize(i,(1280,768))
        #mean,std=cv2.meanStdDev(i)
        #i[:,:,0]=(i[:,:,0]-mean[0])/std[0]
        #i[:,:,1]=(i[:,:,1]-mean[1])/std[1]
        #i[:,:,2]=(i[:,:,2]-mean[2])/std[2]
        temp_img.append(i)
    return temp_img


rospy.init_node("test_node")

model=load_model('/home/zaferpc/abb_ws/src/Disassembly-Perception/src/perception/models/segmentation_model.h5')
model.summary()
imgs = load_images("/home/zaferpc/data/laptop_components/exp_auto_direct_light/imgs")
cv2.imshow("img", imgs[0])
cv2.waitKey(0)

test_image = model.predict(np.array(imgs))

cv2.imshow("test_img", test_image[0])
cv2.waitKey(0)

for i in range(len(test_image)):
    imgray = cv2.cvtColor(test_image[i], cv2.COLOR_BGR2GRAY)
    draw_on = np.copy(imgs[i])

    norm_imgray = cv2.normalize(imgray,None,0,255,cv2.NORM_MINMAX,cv2.CV_32F)

    ret2,thresh_imgray = cv2.threshold(np.uint8(norm_imgray),150,255,cv2.THRESH_BINARY)
    filtered_imgray = cv2.medianBlur(thresh_imgray, 11, 0)
    filtered_imgray = cv2.GaussianBlur(filtered_imgray, (11, 11), 0)
    #imgray4 = cv2.Canny(imgray4,100,200)
    #imgray4 = cv2.adaptiveThreshold(np.uint8(imgray3),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    contours, hierarchy = cv2.findContours(filtered_imgray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(draw_on, contours, -1, (0,255,0), 5)

    cv2.imshow("result", draw_on)
    cv2.waitKey(0)

cv2.destroyAllWindows()