# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 09:41:10 2022

@author: scc
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.chdir ('C:/Users/scc/Desktop/GATA/data_class_seperated/')

dir_train='C:/Users/scc/Desktop/GATA/data_class_seperated/train'
dir_val1=  'C:/Users/scc/Desktop/GATA/data_class_seperated/validation'
traingen = ImageDataGenerator(
rotation_range=30,
width_shift_range=0.2,
height_shift_range=0.2,
horizontal_flip=True, rescale=1./255
)
batch_size = 20
train_generator = traingen.flow_from_directory(
    directory=dir_train,
    target_size=(32, 32),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    subset='training',
    shuffle=True,
    seed=42   )
import tensorflow as tf

valid_generator = ImageDataGenerator(rescale=1/255.)

#train_generator = valgene.flow_from_directory(directory='Val_class', target_size=(100, 100), color_mode="rgb", batch_size=batch_size, class_mode="categorical",  subset='validation', shuffle=True,  seed=42)
valid_generator.flow_from_directory(directory=dir_val1, subset='validation', color_mode="grayscale",

                           target_size=(32, 32),
                           class_mode='categorical',
                           batch_size=32, shuffle=False)
os.chdir ('C:/Users/scc/Desktop/GATA/')
def model_import(raw=False):
    if raw:
        from raw_model_tested import my_model_raw
        my_model=my_model_raw
    
    else:
        from model_regularized import my_model_regularized
        my_model=my_model_regularized
    return my_model
my_model=model_import(False)
#os.chdir ('C:/Users/scc/Desktop/GATA/data_class_seperated/')

opt=tf.keras.optimizers.Adam(0.001, beta_1=0.9 )

#loss=tf.keras.losses.categorical_crossentropy()

my_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
my_model.fit(
        train_generator,
        steps_per_epoch=7623 // batch_size,
        epochs=8,
        validation_data=valid_generator,
        validation_steps=250 // batch_size)