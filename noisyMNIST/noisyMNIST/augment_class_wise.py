# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 08:45:56 2022

@author: scc
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 06:19:07 2022

@author: scc
"""
'''
import Augmentor
def aug_images(path):
    #dir_='C:/Users/scc/Desktop/GATA/data_class_seperated/train'
    p = Augmentor.Pipeline(path)
    p.rotate(probability=0.2, max_left_rotation=15, max_right_rotation=15)
    p.zoom(probability=0.05, min_factor=1.1, max_factor=1.5)
    p.random_distortion(probability=0.2, grid_width=2, grid_height=2, magnitude=2)
    augmented_images = p.sample(800)
import os
dir_='C:/Users/scc/Desktop/GATA/data_class_seperated/train'
aug_need=['2','5','8']
for i in range(len(aug_need)):
    paths=os.path.join(dir_, aug_need[i])
    aug_images(paths)


Images=[]
labels=[]
for i in range(len(aug_need)):
    A=os.listdir(os.path.join(aug_need[i]))
    for i , _ in enumerate(data_list):
      #  Img=cv2.resize(cv2.imread(os.path.join(data_dir,data_list[i]), 0),(20,32)) #wait for resize!!!
        Img=cv2.imread(os.path.join(data_dir,data_list[i]), 0)
        Image_.append(Img)
        label, _ =data_list[i].split('.')[0].split('_')
        lab=np.uint32(label)
        im_label.append(lab)
    return Image_, np.array(im_label) , #np.array(Image_)
    

dir_='C:/Users/scc/Desktop/train'
os.listdir(dir)

'''
#########

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
absolute_path = os.path.dirname(__file__)
absolute_path
#change directory 
dir_='train_augmented'


A=os.listdir(dir_)

AA=[os.path.join(dir_,i) for i in A]
#print(AA)
images=[]
labels=[]
#class_images=os.listdir(AA[1])
#print(class_images)


for i,k in enumerate(AA):
    class_images=os.listdir(AA[i])
    for j,img  in enumerate(class_images):
        Z=os.path.join(AA[i],img)
        Img=cv2.imread(Z, 0)
        Img=cv2.resize(Img, (32,32))

        images.append(Img)
        labels.append(i)
#print(len(images), len(labels))
Images=np.array(images)
Labels=np.array(np.uint32(labels))
