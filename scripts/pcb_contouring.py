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
from sensor_msgs.msg import Image
import ros_numpy
from scipy.signal import savgol_filter
from std_msgs.msg import Float32MultiArray


from tensorflow.python.keras import backend as K
from tensorflow.keras.utils import CustomObjectScope
def relu6(x):
  return K.relu(x, max_value=6)
# with CustomObjectScope({'relu6': relu6}):
#     keras_mobilenet= tf.keras.applications.mobilenet.MobileNet(weights=None)
#     keras_mobilenet.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
#                           loss='categorical_crossentropy',
#                           metrics=['accuracy'])
#     mobilenet_estimator = tf.keras.estimator.model_to_estimator(keras_model=keras_mobilenet)

def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

def load_images(path):
    temp_img, temp_mask = [], []
    images = glob(os.path.join(path, '*.jpg'))
    # images=images.sort()
    for i in tqdm(sorted(images)):
        i = cv2.imread(i)
        # i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        i = cv2.normalize(i, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        i = cv2.resize(i, (1280, 768))
        # mean,std=cv2.meanStdDev(i)
        # i[:,:,0]=(i[:,:,0]-mean[0])/std[0]
        # i[:,:,1]=(i[:,:,1]-mean[1])/std[1]
        # i[:,:,2]=(i[:,:,2]-mean[2])/std[2]
        temp_img.append(i)
    return temp_img


rospy.init_node("test_node")
message_pub = rospy.Publisher("/cutting_path",Float32MultiArray)

with CustomObjectScope({'relu6': relu6}):
    model = load_model(
    '/home/zaferpc/abb_ws/src/perception/models/segmentation_model_deepLab.h5')
model.summary()
#imgs = load_images("/home/zaferpc/data/laptop_components/exp_auto_direct_light/imgs")
image_raw = rospy.wait_for_message('/camera/color/image_raw',Image)
image_captured = ros_numpy.numpify(image_raw)
image_captured = cv2.cvtColor(image_captured, cv2.COLOR_BGR2RGB)
image_captured = cv2.normalize(image_captured, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
image_captured = cv2.resize(image_captured, (1280, 768))
imgs = [image_captured]
cv2.imshow("img", imgs[0])
#cv2.waitKey(0)

test_image = model.predict(np.array(imgs))

cv2.imshow("test_img", test_image[0])
#cv2.waitKey(0)

for i in range(len(test_image)):
    imgray = cv2.cvtColor(test_image[i], cv2.COLOR_BGR2GRAY)
    draw_on = np.copy(imgs[i])

    norm_imgray = cv2.normalize(
        imgray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)

    ret2, thresh_imgray = cv2.threshold(
        np.uint8(norm_imgray), 150, 255, cv2.THRESH_BINARY)
    filtered_imgray = cv2.medianBlur(thresh_imgray, 11, 0)
    filtered_imgray = cv2.GaussianBlur(filtered_imgray, (11, 11), 0)
    filtered_imgray = cv2.resize(filtered_imgray,(1280,720))
    cv2.imshow("filtered_img", filtered_imgray)
#    cv2.waitKey(0)
    #imgray4 = cv2.Canny(imgray4,100,200)
    #imgray4 = cv2.adaptiveThreshold(np.uint8(imgray3),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    contours, hierarchy = cv2.findContours(
        filtered_imgray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x),reverse=True)
    cnt2 = cntsSorted[0]
    cnt2[:,0,0] = savgol_filter(cnt2[:,0,0],151,3)
    cnt2[:,0,1] = savgol_filter(cnt2[:,0,1],151,3)
    cnt_resized=scale_contour(cnt2,0.9)
    print(cnt_resized)
    cv2.drawContours(draw_on, cnt_resized, -1, (0, 255, 0), 5)
    
    for i in range(len(cnt_resized) - 1):
        cv2.line(draw_on, (cnt_resized[i][0][0], cnt_resized[i][0][1]), (cnt_resized[i+1][0][0], cnt_resized[i+1][0][1]), (0, 0, 255), 2)
        
    cv2.imshow("result", draw_on)
    cv2.waitKey(0)
    msg = Float32MultiArray()
    msg.data=[]

    for cnt_counter in cnt_resized:
        msg.data.append(cnt_counter[0][1])
        msg.data.append(cnt_counter[0][0])
    message_pub.publish(msg)

cv2.destroyAllWindows()
