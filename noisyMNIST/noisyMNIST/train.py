import numpy as np
import cv2
import time, datetime
import matplotlib.pyplot as plt

from sklearn.utils import class_weight 
import tensorflow as tf
#tf.test.is_gpu_available()
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Input, DepthwiseConv2D, add , Flatten, GlobalAveragePooling2D , UpSampling2D , Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import TensorBoard
from data_load import image_label_data , class_weight

from utils import shuffle_data , norrmalize_data, plot_learning_curves , zero_centered, model_import
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import load_model


#%% callbacks
filepath='saved_model/weights-improvement-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5'
checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath,
                                              verbose=1,
                                              save_best_only=True)
earlystop=tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=30 ,
                                           verbose=1 )
lr_callback=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 patience=5, 
                                                 factor=0.05,
                                                 min_delta=1e-2)
tbCallBack=TensorBoard(log_dir='.\my_logs', histogram_freq=0,  write_graph=True, write_images=True)
#model.save('.h5')
'''
epochs=50
lr=0.1
decay=lr/epochs
momentum=0.8
def exp_decay(epoch):
    lrate=lr*np.exp(-decay*epoch)
    return lrate
#lr_callback=tf.keras.callbacks.LearningRateScheduler(schedule=(lambda epoch: 1e-4 * 10**(epoch/20)))
#lr_callback=tf.keras.callbacks.LearningRateScheduler(schedule=exp_decay)
only_resize=True
'''
#name="fariba-{}".format(int(time.time()))
#tensorboard=TensorBoard(log_dir='.logs/{}'.format(name))
#logdir="logs1/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
np.random.seed(11)
tf.random.set_seed(11)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# loading data and label from data_load.py in different way 

train_data , train_label,val_data, val_label =image_label_data(only_resize=False,
                                                               only_resize_and_balance=False, 
                                                               preprocess_and_resize=False,
                                                               preprocess_and_resize_and_balanced=False,
                                                               augment_class_wise_=False,
                                                               augment_class_wise_preprocess=True)
print('train_data.shape, val_data.shape, train_label.shape, val_label.shape :\n')
print(train_data.shape, val_data.shape, train_label.shape, val_label.shape)
#print(train_data.max(), val_data.max(), train_label.max(), val_label.max())
#print(train_data.min(), val_data.min(), train_label.min(), val_label.min())
# normalize (./255 and float)
train_data, train_label = norrmalize_data(train_data, train_label)
val_data , val_label    = norrmalize_data(val_data, val_label)
#  shuffle
train_data, train_label =  shuffle_data(train_data, train_label)



#%%
# choose between raw or regularized model
my_model=model_import(raw=False)
# function for compile and fit with different options
def model_fit_compile( my_model,
                      fit_by_aug=False,
                      fit_by_focal_loss=False,
                      normal_fit=False,
                      fit_by_class_weight=False,
                      fit_KFcrossvalidation=False,
                      batch_size=32,                      
                      epochs=100):
    import tensorflow as tf
    if normal_fit:
       # metric=tf.keras.metrics.Accuracy()#CategoricalAccuracy()
        #metric=tf.keras.metrics.Precision()
        opt=tf.keras.optimizers.Adam(0.001, beta_1=0.9 )
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        metric=tf.keras.metrics.CategoricalAccuracy()
        
        my_model.compile(loss=loss, 
                         optimizer=opt, 
                         metrics=metric)
        history=my_model.fit(train_data ,
                             train_label ,
                             batch_size=batch_size,
                             epochs=epochs,
                             validation_data=(val_data,val_label),
                             callbacks=[checkpoint,tbCallBack, lr_callback,earlystop])
        return history
    elif fit_by_focal_loss: 
        #for imbalance data this loss can be used 
        from tensorflow.keras import backend as K
        import tensorflow as tf
        def focal_loss(gamma=2., alpha=.25):
            def focal_loss_fixed(y_true, y_pred):
                pt_1 =tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
                pt_0= tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
                return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
            return focal_loss_fixed

        loss=[focal_loss(alpha=.25, gamma=2)]
        #opt=tf.keras.optimizers.SGD(0.01)
        opt=tf.keras.optimizers.Adam(0.001, beta_1=0.9 )
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        metric=tf.keras.metrics.CategoricalAccuracy()
        my_model=tf.keras.Sequential([my_model,
                                      tf.keras.layers.Softmax()]) #add softmax to model

        
        my_model.compile(loss=[focal_loss(alpha=.25, gamma=2)],
                         optimizer=opt,
                         metrics=metric)#tf.keras.metrics.CategoricalAccuracy())#['accuracy'])

        history=my_model.fit(train_data,
                             train_label,
                             batch_size=batch_size, 
                             epochs=epochs,
                             validation_data=(val_data,val_label), callbacks=[checkpoint,tbCallBack, lr_callback,earlystop])
        return history
    elif fit_by_class_weight:
        from sklearn.utils import class_weight
 
        opt=tf.keras.optimizers.Adam(0.001, beta_1=0.9 )
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        metric=tf.keras.metrics.CategoricalAccuracy()
#        opt=tf.keras.optimizers.SGD(0.1)

        my_model.compile(loss=loss,
                         optimizer=opt,
                         metrics=metric)#'categorical_crossentropy'
         
        class_weights= {0:0.7623,
                        1: 0.7623,
                        2:3.66490385,
                        3: 0.7623,
                        4: 0.7623,
                        5: 3.75517241,
                        6:0.7623,
                        7:0.7623,
                        8: 3.59575472,
                        9:0.7623}
        manual_class_weight={0:1, 1: 1 ,2: 5,3: 1,4: 1,5: 5,6:1,7:1,8: 5,9:1}
        history=my_model.fit(train_data ,
                             train_label,
                             batch_size=batch_size, 
                             epochs=epochs,
                             validation_data=(val_data,val_label),                             
                             # by sklearn we get float number
                             # I set it also manually to be integer( i read they lead to different results)
                             
                             #class_weight=manual_class_weight,
                             class_weight= manual_class_weight,
                             callbacks=[checkpoint,tbCallBack, lr_callback,earlystop])
        return history
    elif fit_KFcrossvalidation:
        train_concat=np.concatenate((train_data, val_data), axis=0)
      #  train_concat=train_data
        #label_concat=train_label
        label_concat=np.concatenate((train_label, val_label), axis=0)

        from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
        #StratifiedKFold.
 #       from keras.wrappers.scikit_learn import KerasClassifier
  #      def build_classifier():
   #         return final_model()
    #    classifier=KerasClassifier(build_fn=build_classifier, batch_size=64, epochs=100)
     #   accuracies=cross_val_Score(estimator=classifier, X=train_data, y=train_label, cv=10, n_jobs=-1) #or train_concat?!
      #  print('accuracies:{}'.format(np.mean(accuracies)))
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=35)
        #.split(train_concat, label_concat)
        scores = []
        acc_per_fold = []
        loss_per_fold = []
        fold_no=1
        for train_ix, test_ix in kfold.split(train_concat, np.argmax(label_concat,axis=1)):
            train_X , test_X= train_concat[train_ix] , train_concat[test_ix]
            train_y , test_y= label_concat[train_ix],label_concat[test_ix]
            my_model=model_import(raw=False)

            opt=tf.keras.optimizers.Adam(0.001, beta_1=0.9 )
            #opt=tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.001/10, nesterov=False)
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            metric=tf.keras.metrics.CategoricalAccuracy()
            my_model.compile(loss=loss, optimizer=opt, metrics=[metric])#['accuracy']) 
            #    history= my_model_regularized.fit(train_data ,train_label ,batch_size=100, epochs=50,validation_split=0.1,  class_weight={0:1, 1: 1 ,2: 5,3: 1,4: 1,5: 5,6:1,7:1,8: 5,9:1})## validation_data=(val_data,val_label))
            history=my_model.fit(train_X ,train_y,batch_size=32, epochs=100, validation_data=(test_X,test_y),
                                 #class_weight={0:1, 1: 1 ,2: 5,3: 1,4: 1,5: 5,6:1,7:1,8: 5,9:1})
                                  callbacks=[checkpoint,tbCallBack, lr_callback,earlystop])
            plot_learning_curves(history)

            scores = my_model.evaluate(test_X, test_y, verbose=0)
            print(f'Score for fold {fold_no}: {my_model.metrics_names[0]} of {scores[0]}; {my_model.metrics_names[1]} of {scores[1]*100}%')
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])

              # Increase fold number
            fold_no = fold_no + 1
    
            score = my_model.evaluate(val_data,val_label, verbose=1)
            print("\nTest Loss:", score[0])
            print('Test accuracy:', score[1])
            score = my_model.evaluate(train_data,train_label, verbose=1)
            print('Train loss:', score[0])
            print('Train accuracy:', score[1])
    
        return history 
    elif fit_by_aug:
        train_datagen = ImageDataGenerator(rescale=1)#,
                                           #zoom_range=0.03,
                                           #width_shift_range=0.008)#, rotation_range=0.15, brightness_range=(0.01,0.04))
        val_datagen =  ImageDataGenerator(rescale=1)
        
        train_generator      = train_datagen.flow(train_data,train_label,batch_size=64, shuffle=True)
        validation_generator = val_datagen.flow(val_data,val_label, batch_size=64, shuffle=False)
        
        opt=tf.keras.optimizers.Adam(0.002, beta_1=0.9)
      #  opt=tf.keras.optimizers.RMSprop()
        #opt=tf.keras.optimizers.Adam(0.001, beta_1=0.9 )
      #  opt=tf.keras.optimizers.SGD(0.01, momentum=0.9)
        #loss = tf.keras.losses.SparseCategoricalCrossentropy()#sparse_categorical_crossentropy()#(y_true, y_pred)CategoricalCrossentropy(from_logits=True)
        metric=tf.keras.metrics.CategoricalAccuracy()
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

#           loss = tf.keras.losses.Hinge()#CategoricalCrossentropy(from_logits=True)
#
 #       my_model=tf.keras.Sequential([my_model,
  #                                    tf.keras.layers.Softmax()])'''
        my_model.compile(loss=loss,
                         optimizer=opt,
                         metrics=metric)
#        batch_size=64
        history=my_model.fit(train_generator,
                            steps_per_epoch=len(train_data)// batch_size,
                            epochs=epochs,
                            validation_data=validation_generator,
                            validation_steps=len(val_data) // batch_size,
                            callbacks=[checkpoint,tbCallBack, lr_callback,earlystop])#, class_weight={0:1, 1: 1 ,2: 5,3: 1,4: 1,5: 5,6:1,7:1,8: 5,9:1})
        #score = model.evaluate_generator(valid_generator)
        #predict = model.predict_generator(test_generator)
        ## predict the class label
        #y_classes = predict.argmax(axis=-1)
        #, class_weight={0:1, 1: 1 ,2: 5,3: 1,4: 1,5: 5,6:1,7:1,8: 5,9:1})
        #score = model.evaluate_generator(valid_generator)
        #predict = model.predict_generator(test_generator)
        ## predict the class label
        #y_classes = predict.argmax(axis=-1)

history= model_fit_compile(my_model,
                           fit_by_aug = False,
                           fit_by_focal_loss = False,
                           normal_fit = True,
                           fit_by_class_weight=False,
                           fit_KFcrossvalidation=False,
                           batch_size=64, 
                           epochs=100)

plot_learning_curves(history)
score = my_model.evaluate(val_data,val_label, verbose=1)
print("\nTest Loss:", score[0])
print('Test accuracy:', score[1])
score = my_model.evaluate(train_data,train_label, verbose=1)
print('Train loss:', score[0])
print('Train accuracy:', score[1])

#%%
'''
my_modelsaved= load_model('C:/Users/scc/Desktop/Fariba GATA/saved_model/weights-improvement-02-1.00.hdf5')
my_model_regularized=my_modelsaved
score = my_model_regularized.evaluate(val_data,val_label, verbose=1)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])
score = my_model_regularized.evaluate(train_data,train_label, verbose=1)
history = my_model_regularized.fit(train_data,train_label, verbose=1, validation_data=(val_data,val_label), epochs=20)

print('Train loss:', score[0])
print('Train accuracy:', score[1])
'''
#
'''Test score: 0.18554820120334625
Test accuracy: 0.9520000219345093''' #but val loss<acc loss # i used val data inside train data in cross val
#Train loss: 0.03767971694469452
#Train accuracy: 0.9949117302894592
#%%



#%% try split in train (my first try)
####################
my_model_regularized=my_model

#from model_regularized import my_model_regularized
#my_model_regularized=my_model_raw
testt_by_val_split_on_raw_model=False
if testt_by_val_split_on_raw_model:
  #  from model_regularized import my_model_regularized
#    from raw_model import my_model_raw
   
   # my_model_regularized=my_model_raw
    opt=tf.keras.optimizers.Adam(0.001, beta_1=0.9 )
    #opt=tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.001/10, nesterov=False)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metric=tf.keras.metrics.CategoricalAccuracy()
    my_model_regularized.compile(loss=loss, optimizer=opt, metrics=[metric])#['accuracy']) 
#    history= my_model_regularized.fit(train_data ,train_label ,batch_size=100, epochs=50,validation_split=0.1,  class_weight={0:1, 1: 1 ,2: 5,3: 1,4: 1,5: 5,6:1,7:1,8: 5,9:1})## validation_data=(val_data,val_label))
    history=my_model_regularized.fit(train_data ,
                                     train_label ,
                                     batch_size=64,
                                     epochs=50,
                                     validation_split=0.2)
                                   #  validation_data=(val_data,val_label)) 
#                                     class_weight={0:1, 1: 1 ,2: 5,3: 1,4: 1,5: 5,6:1,7:1,8: 5,9:1}
                                   

  #  my_modelsaved= load_model('C:/Users/scc/Desktop/GATA-final/Models/Rezaei/saved_model/yes_weights-improvement-29-0.91.hdf5')
  #  my_model_regularized=my_modelsaved
    score = my_model_regularized.evaluate(val_data,val_label, verbose=1)
    print("\nTest score:", score[0])
    print('Test accuracy:', score[1])
    plot_learning_curves(history)
    
