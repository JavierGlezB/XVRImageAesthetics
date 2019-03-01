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
            [-1., 1., -1., 1., -1.],
            [1., -1., 1., -1., 1.],
            [-1., 1., 0., 1., -1.],
            [1., -1., 1., -1., 1.],
            [-1., 1., -1., 1., -1.]]
        ).astype(float)

        self.k2_2 = np.array([
            [1., -1., 1., -1., 1.],
            [-1., 1., -1., 1., -1.],
            [1., -1., 0., -1., 1.],
            [-1., 1., -1., 1., -1.],
            [1., -1., 1., -1., 1.]]
        ).astype(float)

        self.k3_1 = np.array([
            [-1., 1., -1., 1., -1., 1., -1.],
            [1., -1., 1., -1., 1., -1., 1.],
            [-1., 1., -1., 1., -1., 1., -1.],
            [1., -1., 1., 0., 1., -1., 1.],
            [-1., 1., -1., 1., -1., 1., -1.],
            [1., -1., 1., -1., 1., -1., 1.],
            [-1., 1., -1., 1., -1., 1., -1.]
        ]).astype(float)

        self.k3_2 = np.array([
            [1., -1., 1., -1., 1., -1., 1.],
            [-1., 1., -1., 1., -1., 1., -1.],
            [1., -1., 1., -1., 1., -1., 1.],
            [-1., 1., -1., 0., -1., 1., -1.],
            [1., -1., 1., -1., 1., -1., 1.],
            [-1., 1., -1., 1., -1., 1., -1.],
            [1., -1., 1., -1., 1., -1., 1.]
        ]).astype(float)

        self.k4_1 = np.array([
            [-1., 1., -1., 1., -1., 1., -1., 1., -1.],
            [1., -1., 1., -1., 1., -1., 1., -1., 1.],
            [-1., 1., -1., 1., -1., 1., -1., 1., -1.],
            [1., -1., 1., -1., 1., -1., 1., -1., 1.],
            [-1., 1., -1., 1., 0., 1., -1., 1., -1.],
            [1., -1., 1., -1., 1., -1., 1., -1., 1.],
            [-1., 1., -1., 1., -1., 1., -1., 1., -1.],
            [1., -1., 1., -1., 1., -1., 1., -1., 1.],
            [-1., 1., -1., 1., -1., 1., -1., 1., -1.]]
        ).astype(float)

        self.k4_2 = np.array([
            [1., -1., 1., -1., 1., -1., 1., -1., 1.],
            [-1., 1., -1., 1., -1., 1., -1., 1., -1.],
            [1., -1., 1., -1., 1., -1., 1., -1., 1.],
            [-1., 1., -1., 1., -1., 1., -1., 1., -1.],
            [1., -1., 1., -1., 0., -1., 1., -1., 1.],
            [-1., 1., -1., 1., -1., 1., -1., 1., -1.],
            [1., -1., 1., -1., 1., -1., 1., -1., 1.],
            [-1., 1., -1., 1., -1., 1., -1., 1., -1.],
            [1., -1., 1., -1., 1., -1., 1., -1., 1.],
        ]).astype(float)

        self.sampling_pattern = np.array([
            [1., 0., 0., 0., 1., 0., 0., 0., 1.],
            [0., 1., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 1., 0., 1., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0., 1., 0., 0., 0.],
            [1., 0., 1., 0., 1., 0., 1., 0., 1.],
            [0., 0., 0., 1., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 1., 0., 1., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 1., 0.],
            [1., 0., 0., 0., 1., 0., 0., 0., 1.], ]
        ).astype(float)

        ch_0, ch_1, ch_2, out, graph = self.convolutions_graph()                        

    def convolutions_graph(self):
        graph = tf.Graph()
        with graph.as_default():

            k1_size = self.k1_1.shape[0]
            k2_size = self.k2_1.shape[0]
            k3_size = self.k3_1.shape[0]
            k4_size = self.k4_1.shape[0]

            ch_0 = tf.placeholder(
                dtype=tf.float32, shape=(1, None, None, 1), name='c1')
            ch_1 = tf.placeholder(
                dtype=tf.float32, shape=(1, None, None, 1), name='c2')
            ch_2 = tf.placeholder(
                dtype=tf.float32, shape=(1, None, None, 1), name='c3')

            chanels = [ch_0, ch_1, ch_2]

            k11 = tf.constant(
                self.k1_1,
                dtype=tf.float32,
                shape=(k1_size, k1_size, 1, 1))

            k12 = tf.constant(
                self.k1_2,
                dtype=tf.float32,
                shape=(k1_size, k1_size, 1, 1))

            k21 = tf.constant(
                self.k2_1,
                dtype=tf.float32,
                shape=(k2_size, k2_size, 1, 1))

            k22 = tf.constant(
                self.k2_2,
                dtype=tf.float32,
                shape=(k2_size, k2_size, 1, 1))

            k31 = tf.constant(
                self.k3_1,
                dtype=tf.float32,
                shape=(k3_size, k3_size, 1, 1))

            k32 = tf.constant(
                self.k3_2,
                dtype=tf.float32,
                shape=(k3_size, k3_size, 1, 1))

            k41 = tf.constant(
                self.k4_1,
                dtype=tf.float32,
                shape=(k4_size, k4_size, 1, 1))

            k42 = tf.constant(
                self.k4_2,
                dtype=tf.float32,
                shape=(k4_size, k4_size, 1, 1))

            kernels = [[k11, k11], [k21, k22], [k31, k32], [k41, k42]]
            i = 0
            outs = []

            for ch in chanels:
                for k in kernels:

                    output = tf.nn.conv2d(
                        input=ch,
                        filter=k[0],
                        strides=[1, 1, 1, 1],
                        padding='SAME')

                    output = tf.nn.conv2d(
                        input=output,
                        filter=k[1],
                        strides=[1, 1, 1, 1],
                        padding='SAME')

                    outs.append(output)

            out = tf.concat(axis=1, values=outs)  

            self.ch_0 = ch_0
            self.ch_1 = ch_1
            self.ch_2 = ch_2
            self.out = out
            self.graph = graph        
            
        return ch_0, ch_1, ch_2, out, graph

    def run_convolution_graph(self, image):
        
        photos = os.listdir('./CapptuPhotos/')
        
        
        with tf.Session(graph=self.graph) as sess:
                        
            chanel_0 = image[:, :, 0]
            chanel_1 = image[:, :, 1]
            chanel_2 = image[:, :, 2]
            heigth, width , chanles = image.shape

            result = sess.run(
                self.out,
                feed_dict={
                    self.ch_0: chanel_0[np.newaxis, :, :, np.newaxis],
                    self.ch_1: chanel_1[np.newaxis, :, :, np.newaxis],
                    self.ch_2: chanel_2[np.newaxis, :, :, np.newaxis],
                })

            result = result[0,:,:,0]
            filtered = []
            for i in range(12):
                f_image = result[i * heigth : (i+1) * heigth,:]
                filtered.append(f_image)
        return filtered


    def key_point_descriptor(keypoint, image):

        return keypoint

    def fast(self, img):
        fast = cv2.FastFeatureDetector_create()
        kp = fast.detect(img, None)[:500]
        img2 = cv2.drawKeypoints(img, kp, color=(255, 0, 0), outImage=True)
        img2 = cv2.drawKeypoints(img, kp, color=(255, 0, 0), outImage=True)
        # cv2.imwrite(str(random.random())+'.jpg',img2)
        return kp

    def get_fast_key_points(self, convoluted):
        key_points = []
        for image_filtered in convoluted:
            for chanel in image_filtered:
                q = self.fast(chanel)
                key_points += q
        return key_points

    def get_best_keypoints(self, keypoints):
        responses = np.array([key.response for key in keypoints])
        strongest_response_index = np.argsort(-responses)
        strongest_response = [keypoints[i]
                              for i in strongest_response_index[:500]]
        return strongest_response

    def get_descriptors(self, best_keypoints):
        descriptors = []
        for keypoint in best_keypoints:
            descriptors.append(self.key_point_descriptor(keypoint))
        return descriptors

    def main(self, image):
        width, heigth, chanels = image.shape
        convolved = self.convolutions_graph(image)
        return convolved
