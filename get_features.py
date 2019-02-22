import tensorflow as tf
import cv2
import numpy as np 
import os
import matplotlib.pyplot as plt
import random

class Convolve():

    def __init__(self,):
        
        self.k1_1 = np.array([ 
            [-1, 1, -1],
            [1, 0, 1],
            [-1, 1, -1]]).astype(float)  

        self.k1_2 = np.array([ 
            [1, -1, 1],
            [-1, 0, -1],
            [1, -1, 1]]).astype(float)  

        self.k2_1 = np.array([
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 0., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.]]
            ).astype(float)  

        self.k2_2 = np.array([
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 0., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.]]
            ).astype(float)  

        self.k3_1 = np.array([
            [1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 0., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1.]
            ]).astype(float)

        self.k3_2 = np.array([
            [1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 0., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1.]
            ]).astype(float)


        self.k4_1 = np.array([
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 0., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.]]
            ).astype(float)

        self.k4_2 = np.array([
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 0., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1., 1.]]
            ).astype(float)


        self.sampling_pattern = np.array([
            [1., 0., 0., 0., 1., 0., 0., 0., 1.],
            [0., 1., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 1., 0., 1., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0., 1., 0., 0., 0.],
            [1., 0., 1., 0., 1., 0., 1., 0., 1.],
            [0., 0., 0., 1., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 1., 0., 1., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 1., 0.],
            [1., 0., 0., 0., 1., 0., 0., 0., 1.],]
            ).astype(float)
    
    def convolve_image(self,image, k1,k2):
        
        input_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(1, image.shape[0], image.shape[1], 1))

        kernel_size = k1.shape[0]

        with tf.name_scope('convolution'):
            k1 = tf.constant( 
                k1, 
                dtype=tf.float32, 
                shape=(kernel_size, kernel_size, 1, 1))

            k2 = tf.constant(
                k2, 
                dtype=tf.float32, 
                shape=(kernel_size, kernel_size, 1, 1))

            output = tf.nn.conv2d(
                input=input_placeholder, 
                filter=k1, 
                strides=[1, 1, 1, 1], 
                padding='SAME')

            output = tf.nn.conv2d(
                input=output, 
                filter=k2, 
                strides=[1, 1, 1, 1], 
                padding='SAME')
        
        chanel_0 = image[:,:,0]
        chanel_1 = image[:,:,1]
        chanel_2 = image[:,:,2]

        with tf.Session() as sess:
            result_0 = sess.run(output, feed_dict={
                input_placeholder: chanel_0[np.newaxis, :, :, np.newaxis]})
            result_1 = sess.run(output, feed_dict={
                input_placeholder: chanel_1[np.newaxis, :, :, np.newaxis]})
            result_2 = sess.run(output, feed_dict={
                input_placeholder: chanel_2[np.newaxis, :, :, np.newaxis]})
    
        return [chanel_0,chanel_1,chanel_1]
    

    def get_convoluted(self, image):
        self.image = image
        res_f1 = self.convolve_image(image,self.k1_1,self.k1_2)
        res_f2 = self.convolve_image(image,self.k2_1,self.k2_2)
        res_f3 = self.convolve_image(image,self.k3_1,self.k3_2)
        res_f4 = self.convolve_image(image,self.k4_1,self.k4_2)
        return [res_f1, res_f2, res_f3, res_f4]

    def get_descriptors(self,convoluted):
        key_points = []
        for image_filtered in convoluted:
            for chanel in image_filtered:
                q = self.fast(chanel)
                key_points.append(q)
        return key_points

       

    def fast(self,img):
        fast = cv2.FastFeatureDetector_create()
        kp = fast.detect(img,None)[:500]
        img2 = cv2.drawKeypoints(img, kp, color=(255,0,0), outImage=True)
        img2 = cv2.drawKeypoints(img, kp, color=(255,0,0), outImage=True)
     	cv2.imwrite(str(random.random())+'.jpg',img2)
        return kp


    def main(self,image):
        res = self.get_convoluted(image)
        keypoints = self.get_descriptors(res)
        return keypoints


        


