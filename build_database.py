import tensorflow as tf
import pandas as pd
import numpy as np 
import csv
import os
import h5py

DESCRIPTOR_SIZE = 6000

def Shanon(A):
    pA = A / (np.expand_dims(np.nansum(A, axis=1),axis=1) * np.ones(np.shape(A)))
    Shannon2 = -np.nansum(pA*np.log2(pA), axis=1 )
    return Shannon2  

def to_complete_info(mean, e, std):

    missing = DESCRIPTOR_SIZE - len(mean)
    mean_stat = [np.mean(mean), np.std(mean)]
    std_stat = [np.mean(std), np.std(std) ]
    e_stat = [np.mean(e), np.std(e)]

    missing_mean = np.random.normal(mean_stat[0], mean_stat[1], missing)
    missing_std = np.random.normal(std_stat[0], std_stat[1], missing)
    missing_e = np.random.normal(e_stat[0], e_stat[1], missing)
    return missing_mean, missing_std, missing_e

def merge_info(mean, e, std):
    pattern = []
    for i in range(DESCRIPTOR_SIZE):
        pattern += [mean[i], std[i], e[i]]
    return np.array(pattern)

def concatenate_info(mean, std, e):
    return np.concatenate((mean, std, e))
    

def create_db(mode='merge' ):
    kp_path = './kp_test/'
    db_path = './database/'
    
    f = h5py.File(db_path + 'database.h5', mode='w')

    kp_files = os.listdir(kp_path)

    dataset = f.create_dataset(shape=(len(kp_files), DESCRIPTOR_SIZE*3), name='data')

    for i, kp in enumerate(kp_files):

        descriptors = pd.read_csv(kp_path+kp).get_values()    
        mean = np.nanmean(descriptors,axis=1)    
        std = np.nanstd(descriptors,axis=1)    
        e = Shanon(descriptors)
        if len(descriptors) < DESCRIPTOR_SIZE:
            missing_mean, missing_std, missing_e = to_complete_info(mean, e, std)
            mean = np.concatenate((mean, missing_mean))        
            std = np.concatenate((std, missing_std))
            e = np.concatenate((e ,missing_e))
        if mode == 'merge':
            pattern = merge_info(mean, std, e)
        else:
            pattern = concatenate_info(mean, std, e)
        
        dataset[i] = pattern

        print "Progress: {0}% ".format(round(100.*i / float(len(kp_files))),2)

    f.close()
    print "Database was created properly"

