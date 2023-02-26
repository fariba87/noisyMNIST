
# Note: i could not grouped together the model files in model directory because when i did so (as I had a lambda layer in 
# my model architecture ) the Inference class doesnt know my lambda layer 
# i tried stackoverflow, there has benn similar problems, but they said as i move to same directory it can work !!
# so sorry for this problem :)


import tensorflow as tf
import numpy as np
import os
import cv2
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from Image_Preprocessing import preprocessing
from data_load import x_val, y_val, x_train , y_train
#from utils import my_Relu6  #model.utils import my_Relu6
from tensorflow.keras.models import load_model
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
#%% i tried several model and save the best checkpoint for testing
#model_path='D:/GATA-fariba rezaei/Models/Rezaei/saved_model/'
absolute_path = os.path.dirname(__file__)
absolute_path
model_path=os.path.join(absolute_path, 'saved_model/')

'''def learning(test_samples):
    model = load_model('C:/Users/scc/Desktop/Fariba GATA/Models/Rezaei/model/weights-improvement-02-1.00.hdf5')'''
    
    
class Inference:
    def __init__(self, model_path, batch_size=16):
        super().__init__()
         
         # model_name='model.h5',
        self.model_path = model_path
        #self.model = tf.keras.models.load_model(self.model_path + 'weights-improvement-16-0.99.hdf5',custom_objects={'tf': tf})  #97.2
        
        self.model = tf.keras.models.load_model(self.model_path + 'weights-improvement-19-0.99.hdf5',custom_objects={'tf': tf})  #97.2

        
        #  self.model = tf.keras.models.load_model(weights-improvement-02-1.00.hdf5')
        self.batch_size = batch_size

    def getScore(self, Xtestlist):
        # self.Xtestlist=Xtestlist
        image_list = []
        for i, img in enumerate(Xtestlist):
            self.img1 = img
            #     self.img1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            #   (img, cv2. BG)
            image = self.preprocessing(self.img1)
            image_list.append(image)
        xtest_numpy = np.array(image_list)
        
        #prediction 
        Yprob = self.model.predict(xtest_numpy)
        Ytesthat = np.argmax(Yprob, axis=1)
        print(Yprob)
        return Yprob

    def preprocessing(self, image1):
        # same as training process
        # a kernel to sharp images (with enhancement)
        # reference: Book: "OpenCV 3 Computer Vision with Python Cookbook" By Alexey Spizhevoy
        kernel_sharpen = np.array([[-1, -1, -1, -1, -1],
                               [-1, 2, 2, 2, -1],
                               [-1, 2, 8, 2, -1],
                               [-1, 2, 2, 2, -1],
                               [-1, -1, -1, -1, -1]]) / 8.0

        image1 = image1  # self.img1
        image2 = cv2.filter2D(image1, -1, kernel_sharpen)
    # median to remove noise
        image3 = cv2.medianBlur(image2, 3)
        ksize, sigma_color, sigma_space = 5, 5, 7
        # bilateral filter is better that gaussian in smoothing while saving details
        image4 = cv2.bilateralFilter(image3, ksize, sigma_color, sigma_space)
        image5 = cv2.resize(image4, (32, 32), interpolation=cv2.INTER_LINEAR)
        image5=image5/255
        image5=image5.astype(np.float32)
        return image5


# input to getscore : Xtest   (i tested it on my val data and get 96%)
Xtest = x_val
ytest = y_val#train#np.argmax(val_label, axis=1) # groundtruth


A = Inference(model_path=model_path, batch_size=16)
Ytest= A.getScore(Xtest)  

Ytesthat = np.argmax(Ytest, axis=1)  # predicted

#Y=my_model.predict(val_data)#,batch_size=16)
#Ytesthat=np.argmax(Y,axis=1)
#ytest_ = np.argmax(Ytest_, axis=1)
precision = precision_score(ytest , Ytesthat ,average='micro')
print('Precision: %f' % precision)
f1 = f1_score(ytest , Ytesthat,average='micro')
recall = recall_score(ytest , Ytesthat,average='micro')
print('#######################')
accuracy = accuracy_score(ytest, Ytesthat)
print('Accuracy: %f' % accuracy)
print('#######################')
matrix = confusion_matrix(ytest , Ytesthat)
print(matrix)
'''
for val data:
Precision: 0.972000
#######################
Accuracy: 0.972000
#######################
[[23  0  0  0  0  1  1  0  0  0]
 [ 0 25  0  0  0  0  0  0  0  0]
 [ 0  0 25  0  0  0  0  0  0  0]
 [ 0  0  0 25  0  0  0  0  0  0]
 [ 0  0  0  0 25  0  0  0  0  0]
 [ 0  0  0  0  0 23  2  0  0  0]
 [ 0  0  0  0  1  0 24  0  0  0]
 [ 0  1  0  0  0  0  0 24  0  0]
 [ 0  0  1  0  0  0  0  0 24  0]
 [ 0  0  0  0  0  0  0  0  0 25]]

'''
#

'''
Precision: 0.991080
#######################
Accuracy: 0.991080
#######################
[[ 983    0    0    2    6    0    2    0    5    2]
 [   0  998    0    0    0    1    0    0    0    1]
 [   0    0  206    0    0    0    0    2    0    0]
 [   0    2    0  992    0    3    0    0    2    1]
 [   0    0    0    0 1000    0    0    0    0    0]
 [   0    0    0    0    0  202    0    0    0    1]
 [   7    1    0    0    1   10  979    0    2    0]
 [   0    6    1    0    1    0    0  992    0    0]
 [   0    0    0    1    0    0    1    0  210    0]
 [   0    3    1    0    1    0    0    1    1  993]]
'''
