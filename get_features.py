import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import time
import csv
import math
import infinity
from utils import entropy, entropy_ideal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
class Convolve():

    def __init__(self,image_dim = 1200):
        self.image_dim = image_dim
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

        regions = np.array([[1,1], [-1,1], [1,-1], [-1 ,-1]])
        pattern = np.array([[2,0], [4,0], [0,2], [0,4]] + [[i,i]for i in range(5)])
        pattern = [ (r * pattern).tolist()  for r in regions]
        pos = set()
        for sector in pattern:
            for tupl in sector:
               pos.add(tuple(tupl))
        self.pattern = list(pos)


        self.filtered = []
        self.i = 0

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

                    output_0 = tf.nn.conv2d(
                        input=ch,
                        filter=k[0],
                        strides=[1, 1, 1, 1],
                        padding='SAME')

                    output_1 = tf.nn.conv2d(
                        input=ch,
                        filter=k[1],
                        strides=[1, 1, 1, 1],
                        padding='SAME')

                    output = tf.sqrt(tf.square(output_0) + tf.square(output_1))
                    outs.append(output)

            out = tf.concat(axis=1, values=outs)  

            self.ch_0 = ch_0
            self.ch_1 = ch_1
            self.ch_2 = ch_2
            self.out = out
            self.graph = graph        
            
        return ch_0, ch_1, ch_2, out, graph

    def run_convolution_graph(self, image):
        
        
                
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
                filtered.append(  (255 * (f_image / np.max(f_image))).astype('uint8') )
            self.filtered = filtered
        return filtered

    def fast(self, img):
        fast = cv2.FastFeatureDetector_create()
        #fast.setNonmaxSuppression(0)  // non max supression didnt work
        kp = fast.detect(img, None)
        best = self.get_best_keypoints(kp)
        #img2 = cv2.drawKeypoints(img, best, color=(255, 0, 0), outImage=True)# save kp image
        #cv2.imwrite('./bestkeypointsImages/'+str(self.i)+'.jpg',img2)
        #self.i += 1
        return best

    def get_best_keypoints(self, keypoints, max_number=500):
        responses = np.array([key.response for key in keypoints])
        strongest_response_index = np.argsort(-responses)
        strongest_response = [keypoints[i]
                              for i in strongest_response_index[:max_number]]
        return strongest_response

    def get_fast_kp(self,filtered):
        key_points = []
        for image in filtered:
            kp = self.fast(image)
            key_points.append(kp)
        return key_points#self.get_best_keypoints(key_points)

    def get_descriptors(self, best_keypoints):
        descriptors = []
        for f_index, image_kp in enumerate(best_keypoints):
            for kp in image_kp:
                descriptor = self.valid_pattern(kp, f_index)
                #mean = np.mean(descriptor)
                #std = np.std(descriptor)
                #entropy = entropy(descriptor)
                #var_hist = np.var(descriptor)
                #descriptors.append([mean, std, entropy, var_hist])
                descriptors.append(descriptor)
        return descriptors


    def valid_pattern(self, kp, f_index):
        valid = []
        x, y = kp.pt
        for pattern_x, pattern_y in self.pattern:
            new_x, new_y = ( int(x + pattern_x), int(y + pattern_y)) 
            if  ((new_x >= 0 and new_x < self.image_dim) and
            (new_y >= 0 and new_y < self.image_dim)):
                diff = self.neighbours_difference(new_x, new_y, f_index)
                valid.append(diff)
        return valid


    def neighbours_difference(self, x, y, f_index):
        valid_n = []
        for i in range(-1,2):
            for j in range(-1,2):
                if i != j:
                    x_n = x - i
                    y_n = y + j 
                    if ((x_n >= 0 and x_n < self.image_dim) and 
                        (y_n >= 0 and y_n < self.image_dim)):
                        valid_n.append(self.filtered[f_index][x_n,y_n])
        n_mean = np.mean(valid_n)
        kp_value = self.filtered[f_index][x,y]
        return round(abs(kp_value - n_mean) ,2)

def test(stop = infinity.inf):
    image_path = './CapptuPhotos/'
    images_names = os.listdir('./CapptuPhotos/')
    total_images = len (images_names)
    print "{0} images found".format(total_images)
    image_dim = 1200
    con = Convolve(image_dim= image_dim)
    con.convolutions_graph()
    t1 = time.time()


    for i, im_name in enumerate(images_names):
        file_path = './kp/'+im_name+'.csv'
        with open(file_path, mode='w') as log_file:
            descriptor_writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            print 'Image: ' + im_name  +', Progress: ' + str(100 * i /float(total_images) ) + "%"
            try:
                image = cv2.cvtColor(cv2.resize(cv2.imread(image_path+im_name), (image_dim, image_dim)), cv2.COLOR_RGB2YCrCb)
                filtered = con. run_convolution_graph(image)
                key_points = con.get_fast_kp(filtered)
                #print len(key_points)
                descriptors = con.get_descriptors(key_points)
                #print len(descriptors)
                for descriptor in descriptors:
                    descriptor_writer.writerow(descriptor)
                erase = False
            except:
                #pass
                erase = True
                #descriptor_writer.writerow(['[Fail]',i,im_name])
        if erase == True:
            os.remove(file_path)

        if i>=stop:
            break
    t2 = time.time()
    print "Elapsed Time (s): {0}".format(t2-t1)
    #return filtered, image, key_points, descriptors
