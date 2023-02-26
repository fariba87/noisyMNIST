

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import cv2
from tensorflow.keras.utils import to_categorical
#from model_architecture import my_model

#tf.test.is_gpu_available()
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Input, DepthwiseConv2D, add , Flatten, GlobalAveragePooling2D , UpSampling2D , Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#from tensorflow.keras.layers.experimental.
from data_load import image_label_data , class_weight

from utils import shuffle_data , norrmalize_data, plot_learning_curves

#from data_load import preprocessing, y_train , y_val, x_train_preprocess , x_val_preprocess
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import TensorBoard

#from model_regularized import my_model_regularized
#my_model=my_model_regularized

import numpy as np
import cv2

#tf.random.set_seed(1)  
np.random.seed(11)
tf.random.set_seed(11)
########################################################################################
##  path
SEED = 999
np.random.seed(SEED)
# data directory 
data_dir='C:/Users/scc/Desktop/DeepTask-Gata/Data'
train='train'
val='valid'

train_dir=os.path.join(data_dir, train)
val_dir=os.path.join(data_dir, val)
train_list=os.listdir(train_dir)
val_list=os.listdir(val_dir)

paths_train=[os.path.join(train_dir, i) for i in train_list]
paths_val=[os.path.join(val_dir, i) for i in val_list]

#####################################################################################
#  python function to preprocess data. should be applied through tf.py_function as wrapper if we want to map to dataset


########################################################################################
# extraction of images and labels, by tf.image to be used in tf.data pipeline
def load_image_and_label(image_path):#, target_size=(32, 32)):
    classes=['0','1','2','3','4','5','6','7','8','9']

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.rgb_to_grayscale(image)       
    image = tf.image.convert_image_dtype(image, np.float32)
      
   # image=tf.expand_dims(image, axis=-1)
    #
    #image=tf.numpy_function(preprocessing(image))
    #
    image = tf.image.resize(image, (32,32))
    image/=255.
    label = tf.strings.split(image_path, os.path.sep)[-1]#.numpy().decode()
    label=tf.strings.split(label, sep='.')[-2]
    label=tf.strings.split(label, sep='_')[0]
    #label, _ =label.split('.')[0].split('_')
    #label = to_categorical((np.int0(label), 10)) #
    label=(label == classes)  # One-hot encode.
    label = tf.dtypes.cast(label, tf.float32)

    return image, label
########################################################################################
def get_train_val_dataset(preprocess=False, BATCH_SIZE =8 ,BUFFER_SIZE = 7623):
    if preprocess:
        # train and val dataset also preprocess with preprocessing function
        
        ds_train_img=dataset_train.map(lambda a,b:a).map(lambda x: tf.py_function(preprocessing,[x],[tf.float32]))
        ds_train_label=dataset_train.map(lambda a,b:b)
        ds_val_img=dataset_val.map(lambda a,b:a).map(lambda x: tf.py_function(preprocessing,[x],[tf.float32]))
        ds_val_label=dataset_val.map(lambda a,b:b)

        dataset_train = tf.data.Dataset.zip((ds_train_img, ds_train_label)).shuffle(7623).batch(16).prefetch(1000)#.map(augment)
        dataset_val   = tf.data.Dataset.zip((ds_val_img, ds_val_label)).batch(16)#.map(augment)


    else:
        # raw dataset which is normalized and resized without any preprocessing

        dataset_train = tf.data.Dataset.from_tensor_slices(paths_train).map(load_image_and_label).shuffle(7623).batch(16).prefetch(1000)
        dataset_val   = tf.data.Dataset.from_tensor_slices(paths_val).map(load_image_and_label).batch(16)
        
    return dataset_train , dataset_val

dataset_train , dataset_val=get_train_val_dataset(preprocess=False, BATCH_SIZE =8 ,BUFFER_SIZE = 7623)


'''
for image , label in dataset_train.take(1):
    print(image.shape, label)'''


#####################################################################################
# data augmentation through tf.image (subsequentally map to train dataset)

def augment(image, label):
    image = tf.image.resize_with_crop_or_pad(image, 32, 32)
    image = tf.image.random_crop(image, size=(32, 32, 1))
    #image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.resize(image, (32,32))

    return image, label

augmentflag=False
if augmentflag:
    dataset_train=dataset_train.map(augment)
  
    
''' 

import albumentations as A
import cv2 

class ImageDataset(tf.data.Dataset):
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image
    

train_transform = A.Compose([
    A.RandomResizedCrop(224,224),
    A.HorizontalFlip(p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5),
    A.RandomBrightnessContrast (p=0.5),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  #  ToTensorV2(),
])


val_transform = A.Compose([
    A.Resize(256,256),
    A.CenterCrop(224,224),
    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
 #   ToTensorV2(),
])

train_dataset = ImageDataset(images_filepaths='C:/Users/scc/Desktop/GATA/data_class_seperated/train', transform=train_transform)
val_dataset = ImageDataset(images_filepaths='C:/Users/scc/Desktop/GATA/data_class_seperated/validation', transform=val_transform)
  ''' 
from model_regularized import my_model_regularized
my_model=my_model_regularized
#from raw_model import my_model_raw
opt=tf.keras.optimizers.Adam(0.0001, beta_1=0.9 )
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
my_model.compile(loss=loss, optimizer=opt, metrics=['accuracy']) 
history=my_model.fit(dataset_train ,batch_size=32, epochs=10, validation_data=dataset_val, class_weight={0:1, 1: 1 ,2: 5,3: 1,4: 1,5: 5,6:1,7:1,8: 5,9:1})

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


'''import albumentations as A
AUTOTUNE = tf.data.experimental.AUTOTUNE

from albumentations import (
    Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,
    Rotate
)
transforms = Compose([
            Rotate(limit=20),
            RandomBrightness(limit=0.1),
          #  JpegCompression(quality_lower=85, quality_upper=100, p=0.5),
          #  HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            RandomBrightness(limit=0.2, p=0.5), 
        ])
from functools import partial
def process_image(image, label, img_size):
    # cast and normalize image
    image = tf.image.convert_image_dtype(image, tf.float32)
    # apply simple augmentations
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(image,[img_size, img_size])
    return image, label

#ds_tf = dataset_train.map(partial(process_image, img_size=32)).batch(10)#.prefetch(AUTOTUNE)
#ds_tf
def aug_fn(image, img_size):
    data = {"image":image}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]
    aug_img = tf.cast(aug_img/255.0, tf.float32)
    aug_img = tf.image.resize(aug_img, size=[img_size, img_size])
    return aug_img
def process_data(image, label, img_size):
    aug_img = tf.numpy_function(func=aug_fn, inp=[image, img_size], Tout=tf.float32)
    return aug_img, label
ds_alb =dataset_train.map(partial(process_data, img_size=32))#.prefetch(AUTOTUNE)
ds_alb
for i , j in ds_alb.take(1):
    print(i)
    print(j)

def set_shapes(img, label, img_shape=(32,32,1)):
    img.set_shape(img_shape)
    label.set_shape([])
    return img, label
ds_alb = ds_alb.map(set_shapes, num_parallel_calls=AUTOTUNE).batch(32).prefetch(AUTOTUNE)
ds_alb'''

#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy', run_eagerly=True)



#####################################################################################
# more process in data pipeline (batch, pretech, map, shuffle)

#load_image_and_label(paths, target_size=(32, 32))
#PP=lambda x: tf.py_function(preprocessing,x,[tf.float32])
 # I check if image and label are aligned --->yes
'''Alll=[]
Al_label=[]
for image , label in dataset_train.take(4000):
   print('image{}'.format(np.array(image).shape))#, label)#.shape, label.shape)
   Alll.append(np.array(image))
   Al_label.append(label)
   #plt.imshow(imageW)
   #print('label{}'.format(label))
   #W=image
'''   




'''
for image in ds_train_img.take(10):
    print(image)

'''


#dataset_train1=ds_train.batch(BATCH_SIZE).shuffle(buffer_size=BUFFER_SIZE).prefetch(buffer_size=BUFFER_SIZE)

#dataset_val1=ds_val.batch(BATCH_SIZE)#.shuffle(buffer_size=BUFFER_SIZE).prefetch(buffer_size=BUFFER_SIZE)
'''
for image , label in dataset_train.take(10):
    image, label

for image in ds_train_img.take(1):
    image

'''


from raw_model import my_model_raw
opt=tf.keras.optimizers.Adam(0.001, beta_1=0.9 )
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
my_model_raw.compile(loss=loss, optimizer=opt, metrics=['accuracy']) 
history=my_model_raw.fit(ds_alb ,batch_size=16, epochs=10, validation_data=dataset_val)


from model_regularized import my_model_regularized

from tensorflow.keras import backend as K
import tensorflow as tf

# Compatible with tensorflow backend

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
	return focal_loss_fixed

my_model=my_model_regularized

#loss=[focal_loss(alpha=.25, gamma=2)]
#my_model.compile(loss=[focal_loss(alpha=.25, gamma=2)], optimizer=opt, metrics=['accuracy'])
    
    
    
opt=tf.keras.optimizers.Adam(0.001)
my_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
 #history=my_model.fit(x_train, y_train_ ,batch_size=16, epochs=6, validation_data=(x_val, y_val_), 
     #                    class_weight={0:0.2, 1: 0.2 ,2: 0.8,3: 0.2,4: 0.2,5: 0.8,6:0.2,7:0.2,8: 0.8,9:0.2})#, callbacks=[tensorboard_callback])#callbacks)


history=my_model.fit(dataset_train1, epochs=10, validation_data=dataset_val1)#,validation_data=dsval)
def plot_model(history):
    import matplotlib.pyplot as pyplot
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()


#for image , label in ds_train.take(10):
#   print('image{}'.format(np.array(image).shape))#, label)#.shape, label.shape)
#   print('label{}'.format(label))
  # W=image
###########################################################################################
######################################################################################################################################################################################
###########################################################################################
   ###


##
#x_train should be shuffled
''' 
x_train = x_train / 255.
x_test = x_test / 255.
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train = np.reshape(x_train, (x_train.shape[0], 784))
x_test = np.reshape(x_test, (x_test.shape[0], 784))
'''        
# whether upgrade TF-->2.9(in new environment) or use custom activation function (as my TF version is 2.3)
#x=[-1, 2, 3, -4, 7]



###########################################################################################
# model --> model_architecture.py
'''
def my_Relu6(x):
    return tf.keras.activations.relu(x, alpha=0.0, max_value=6)
    #or based on tensor
   # x=np.array(x, dtype=np.float32)
#    X1=np.where(x < 0, 0, x)

    [x=tf.convert_to_tensor(x, dtype=tf.int32)
    X1=tf.where(x < 0, 0, x)
      X2=tf.where(X1<6, X1, 6)
    return X2]
   # X2=np.where(X1<6, X1, 6)
###########################################################################################
  
#my_Relu6(np.array(x))
###########ok
def BlockA(inputs_shape,exp, outputs_shape, resize_A, RESIZE_SHORTCUT=None):
    #functional 
    shortcut=Input(inputs_shape)#shape=(32,32,16))
    x=shortcut
    #x=tf.keras.layers.experimental.preprocessing.Resizing(resize_A,resize_A, interpolation="bilinear", crop_to_aspect_ratio=False)(x)
    x=tf.keras.layers.Lambda(lambda x: tf.image.resize(x,(resize_A,resize_A)))(x)
    
    x=Conv2D(filters=inputs_shape[-1]*exp, kernel_size=(1,1), activation='relu')(x)
    x=BatchNormalization()(x)
    x=DepthwiseConv2D(kernel_size=(3,3), activation='relu')(x)
    x=BatchNormalization()(x)

    y=Conv2D(filters=outputs_shape, kernel_size=(1,1), name='shape_det')(x)
    x=BatchNormalization()(x)

    if not RESIZE_SHORTCUT :
        z= add([shortcut, y])

    else:
        #shortcut1=tf.keras.layers.experimental.preprocessing.Resizing(RESIZE_SHORTCUT,RESIZE_SHORTCUT, interpolation="bilinear", crop_to_aspect_ratio=False)(shortcut)
        shortcut1=tf.keras.layers.Lambda(lambda x: tf.image.resize(x,(RESIZE_SHORTCUT,RESIZE_SHORTCUT)))(shortcut)

        
        z= add([shortcut1, y])

    block_A_model= Model(inputs=shortcut  , outputs=z)
    #block_A_model.summary()
    #plot_model(block_A_model,show_shapes=True,expand_nested=False )
    return block_A_model

######### Ok
def BlockB(inputs_shape,exp, outputs_shape , resize_B=2):
      #Sequential Model
      block_B_model=Sequential()
      block_B_model.add(Conv2D(filters=inputs_shape[-1]*exp, kernel_size=(1,1), 
                               activation=my_Relu6, input_shape=inputs_shape))
      block_B_model.add(BatchNormalization())

      block_B_model.add(UpSampling2D(size=(resize_B,resize_B)))
      block_B_model.add(DepthwiseConv2D(kernel_size=(3,3),activation=my_Relu6))
      block_B_model.add(BatchNormalization())

      block_B_model.add(Conv2D(filters=outputs_shape, kernel_size=(1,1)))   
      block_B_model.add(BatchNormalization())

      return block_B_model

      #block_B_model.summary()
      #plot_model(block_B_model)
      
      
 
#######################################################
# i try this model as subclassing -but as plot_model does not support visualizing subclasses models, i tried functional one,as well
class BlockC_class(tf.keras.Model):
    def __init__(self, activation=my_Relu6,exp=2):
        super().__init__()
        self.activation=activation
        self.exp=exp
        #vali vooroodi inputs to call miad
       # self.conv1=Conv2D(filters=self.exp*inputs.shape[-1], kernel_size=(1,1),activation=self.activation)
        self.dconv1=DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), activation=my_Relu6)
        self.conv2=Conv2D(filters=1, kernel_size=(1,1),activation=None)
    def call(self, inputs, exp):
        #Conv1=self.conv1(inputs)
        Conv1=Conv2D(filters=self.exp*inputs.shape[-1], kernel_size=(1,1),activation=self.activation)
        Dconv1=self.dconv1(Conv1)
        Conv2=self.conv2(Dconv1)
        return Conv2

def BlockC_byclass(inputs, exp):    
    #subclassing Model
    blockc=BlockC_class()
    return blockc(inputs, exp)

def BlockC (inputs_shape,exp, outputs_shape, resize_C):
    CCC=Input(shape=inputs_shape)
    CC= Conv2D(filters=inputs_shape[-1]*exp, kernel_size=(1,1),activation=my_Relu6)(CCC)
    CC=BatchNormalization()(CC)

    # 
    
    CC=DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), activation=my_Relu6)(CC)
    CC=BatchNormalization()(CC)

    #padding='same' didnt work!
    CC=Conv2D(filters=outputs_shape, kernel_size=(1,1))(CC)
    CC=BatchNormalization()(CC)

    CC=tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (resize_C, resize_C)))(CC)

    #CC=Resizing(
    #resize_C, resize_C, interpolation="bilinear", crop_to_aspect_ratio=False)(CC)
    block_c_model= Model(inputs=CCC, outputs=CC)
    #block_c_model.summary()
    #plot_model(block_c_model, show_shapes=True,expand_nested=False )
    return block_c_model

resize_C=[8,4,1]
resize_A=[6, 4]
RESIZE_SHORTCUT=[None , 2]
######################################################################
Resize_C=[8,4,1]
Resize_A=[6, 4]
RESIZE_SHORTCUT1=[None , 2]
def model_blocks(inputs=(16,16,8),iter_stage_2=1, iter_stage_3=1):
  #iter_stage_2, iter_stage_3=1,1
  X1=Input(inputs)#shape=(16,16,8))
  X=BlockC(inputs_shape=(16,16,8), exp=1, outputs_shape=8, resize_C= Resize_C[0] )(X1)
  for i in range(iter_stage_2):
        X=BlockC(inputs_shape=(8,8,8), exp=2, outputs_shape=16, resize_C=Resize_C[1])(X)#X, exp=2, 16)
        XX=BlockA(inputs_shape=(4,4,16), exp=2, outputs_shape=16 , resize_A= Resize_A[0],  RESIZE_SHORTCUT=None)(X) #(X, exp=2, 16)
  for i in range(iter_stage_3):
        XX=BlockC(inputs_shape=(4,4,16), exp=2, outputs_shape=24, resize_C=Resize_C[2])(X)#(X, exp=2)
        XX=BlockA(inputs_shape=(1,1,24), exp=2, outputs_shape=24, resize_A= Resize_A[1],  RESIZE_SHORTCUT=2)(XX)#(X, exp=2)
#X=BlockC(inputs_shape=(8,8,8), exp=2, outputs_shape=16)(X)#X, exp=2, 16)
  Xout=BlockB(inputs_shape=(2,2,24), exp=2, outputs_shape=32)(XX)
  blocks_model= Model(inputs=X1, outputs=Xout)
#blocks_model.summary()
#plot_model(blocks_model, show_shapes=True,expand_nested=False )
  return blocks_model
############################

AA=Input(shape=(32, 32, 1))    
X11=Conv2D(filters=8, kernel_size=(3,3), strides=(2,2), padding='same')(AA) #ouput:15*15*8
X11=BatchNormalization()(X11)

sub_model=model_blocks(inputs=(16,16,8),iter_stage_2=1, iter_stage_3=1)
X2=sub_model(X11)
X2=GlobalAveragePooling2D()(X2)  #with GAP: 8970 parameters
#X2=Flatten()(X2)  #with flatten 9930 parameters
#X2=Dropout(0.2)(X2)
X2=Dense(units=10,activation='softmax')(X2)


my_model=Model(inputs=AA, outputs=X2)
my_model.summary()
plot_model(my_model, show_shapes=True,expand_nested=False )


'''