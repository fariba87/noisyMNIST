# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 01:17:38 2022

@author: scc
"""

import tensorflow as tf
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import cv2
from tensorflow.keras.utils import to_categorical

np.random.seed(11)
tf.random.set_seed(11)
#from augment_class_wise import Images, Labels
import imblearn
from imblearn.over_sampling import SMOTE
import numpy as np
#from Image_Preprocessing import preprocessing

def shuffle_data(train_data_, train_label_):
   # from random import shuffle
    #ind_list = [i for i in range(len(train_data))]
    #shuffle(ind_list)
    ind_list=np.random.permutation(len(train_data_))
    shuffled_img = np.array(train_data_[ind_list, :,:,:])    
    shuffled_label = np.array(train_label_[ind_list,...])

    return shuffled_img, shuffled_label
def preprocessing(img):
    # a kernel to sharp images (with enhancement)
    # reference: Book: "OpenCV 3 Computer Vision with Python Cookbook" By Alexey Spizhevoy
    kernel_sharpen = np.array([[-1 ,-1 ,-1 ,-1 ,-1],
                               [-1 ,2 ,2 ,2 ,-1],
                               [-1 ,2 ,8 ,2 ,-1],
                               [-1 ,2 ,2 ,2 ,-1],
                               [-1 ,-1 ,-1 ,-1 ,-1]]) / 8.0

    image1 =img
    image2 = cv2.filter2D(image1, -1, kernel_sharpen)
    # median to remove noise
    image3 = cv2.medianBlur(image2, 3)
    ksize, sigma_color, sigma_space= 5, 5, 7
    # bilateral filter is better that gaussian in smoothing while saving details
    image4= cv2.bilateralFilter(image3, ksize, sigma_color, sigma_space)
    # resize to (32,32)
    # Seam Carving can also be used for resizing
    # linear intepolation as we are enlarging and it is lighter than cubic
    image5 =cv2.resize(image4, (32 ,32), interpolation = cv2.INTER_LINEAR)

    return image5


def plot_learning_curves(history):
    import matplotlib.pyplot as pyplot
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.grid()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['categorical_accuracy'], label='train')
    pyplot.plot(history.history['val_categorical_accuracy'], label='test')
    pyplot.legend()
    pyplot.grid()
    pyplot.show()
'''
def my_Relu6(x):
    import tensorflow as tf
    return tf.nn.relu6(x)
    #return tf.keras.activations.relu(x, alpha=0.0, max_value=6)
'''
    #or based on tensor
   # x=np.array(x, dtype=np.float32)
#    X1=np.where(x < 0, 0, x)

def norrmalize_data(train_new,y):
    x_train_= train_new/255.#(train_new-np.mean(train_new))/np.std(train_new) #/ 255.
    x_train = x_train_.astype(np.float32)
    return  x_train ,y


def zero_centered(x, train=True):
    train_mean= np.mean (train_data, axis=0)
   # train_std = np.std(train_data, axis=0)
 
    return (x- train_mean)#/train_std
def model_import(raw=True):
    if raw:
        from raw_model import my_model_raw
        my_model = my_model_raw
    else:
        from model_regularized import my_model_regularized
        # from model_reg_ import my_model_regularized
        my_model = my_model_regularized
    return my_model


#def my_Relu6(x):
 #   return tf.keras.activations.relu(x, alpha=0.0, max_value=6)

  #  '''[x=tf.convert_to_tensor(x, dtype=tf.int32)
   # X1=tf.where(x < 0, 0, x)
    #  X2=tf.where(X1<6, X1, 6)
    #return X2]'''