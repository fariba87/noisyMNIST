
# raw model architecture

import tensorflow as tf
# tf.test.is_gpu_available()
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Input, DepthwiseConv2D, add, Flatten, GlobalAveragePooling2D, UpSampling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import cv2
from utils import my_Relu6

tf.random.set_seed(1)

######################################################################
Resize_C = [8, 4, 1]
Resize_A = [6, 4]
RESIZE_SHORTCUT1 = [None, 2]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def BlockC(inputs_shape, exp, outputs_shape, resize_C):
    CCC = Input(shape=inputs_shape)

    CC = Conv2D(filters=inputs_shape[-1] * exp, kernel_size=(1, 1))(CCC)
    CC = tf.keras.layers.Lambda(my_Relu6, name='relu6_1')(CC)

    CC = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same')(CC)
    CC = tf.keras.layers.Lambda(my_Relu6, name='relu6_2')(CC)

    CC = Conv2D(filters=outputs_shape, kernel_size=(1, 1))(CC)

    #### CC=tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (resize_C, resize_C)))(CC)

    block_c_model = Model(inputs=CCC, outputs=CC)
    # block_c_model.summary()
    # plot_model(block_c_model, show_shapes=True,expand_nested=False )
    return block_c_model


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def BlockA(inputs_shape, exp, outputs_shape, resize_A, RESIZE_SHORTCUT=None):
    # functional
    shortcut = Input(inputs_shape)  # shape=(32,32,16))
    x = shortcut
    # x=tf.keras.layers.experimental.preprocessing.Resizing(resize_A,resize_A, interpolation="bilinear", crop_to_aspect_ratio=False)(x)
    # x=tf.keras.layers.Lambda(lambda x: tf.image.resize(x,(resize_A,resize_A)), name='resize')(x)

    x = Conv2D(filters=inputs_shape[-1] * exp, kernel_size=(1, 1))(x)
    x = tf.keras.layers.Lambda(my_Relu6, name='relu6_1')(x)

    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(x)
    x = tf.keras.layers.Lambda(my_Relu6, name='relu6_2')(x)

    y = Conv2D(filters=outputs_shape, kernel_size=(1, 1), name='conv')(x)

    if not RESIZE_SHORTCUT:
        z = add([shortcut, y])

    else:
        shortcut1 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (RESIZE_SHORTCUT, RESIZE_SHORTCUT)))(shortcut)
        z = add([shortcut1, y])

    block_A_model = Model(inputs=shortcut, outputs=z)
    #   block_A_model.summary()
    #  plot_model(block_A_model,show_shapes=True,expand_nested=False )
    return block_A_model


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def BlockB(inputs_shape, exp, outputs_shape, resize_B=2):
    # Sequential Model
    block_B_model = Sequential()
    block_B_model.add(Conv2D(filters=inputs_shape[-1] * exp, kernel_size=(1, 1),
                             input_shape=inputs_shape))
    block_B_model.add(tf.keras.layers.Lambda(lambda x: my_Relu6(x), name='relu6_1'))

    block_B_model.add(UpSampling2D(size=(resize_B, resize_B)))

    block_B_model.add(DepthwiseConv2D(kernel_size=(3, 3)))
    block_B_model.add(tf.keras.layers.Lambda(lambda x: my_Relu6(x), name='relu6_2'))

    block_B_model.add(Conv2D(filters=outputs_shape, kernel_size=(1, 1)))

    return block_B_model


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def model_blocks(inputs=(16, 16, 8), iter_stage_2=1, iter_stage_3=1):
    # iter_stage_2, iter_stage_3=1,1
    X1 = Input(inputs)  # shape=(16,16,8))
    # stage 1
    X = BlockC(inputs_shape=(16, 16, 8), exp=1, outputs_shape=8, resize_C=Resize_C[0])(X1)
    # stage 2
    for i in range(iter_stage_2):
        X = BlockC(inputs_shape=(8, 8, 8), exp=2, outputs_shape=16, resize_C=Resize_C[1])(X)  # X, exp=2, 16)
        XX = BlockA(inputs_shape=(4, 4, 16), exp=2, outputs_shape=16, resize_A=Resize_A[0], RESIZE_SHORTCUT=None)(
            X)  # (X, exp=2, 16)
    # stage 4
    for i in range(iter_stage_3):
        XX = BlockC(inputs_shape=(4, 4, 16), exp=2, outputs_shape=24, resize_C=Resize_C[2])(X)  # (X, exp=2)
        XX = BlockA(inputs_shape=(2, 2, 24), exp=2, outputs_shape=24, resize_A=Resize_A[1], RESIZE_SHORTCUT=None)(
            XX)  # (X, exp=2)
    # post_block
    Xout = BlockB(inputs_shape=(2, 2, 24), exp=2, outputs_shape=32)(XX)

    blocks_model = Model(inputs=X1, outputs=Xout)
    # blocks_model.summary()
    # plot_model(blocks_model, show_shapes=True,expand_nested=False )
    return blocks_model


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def final_model():
    AA = Input(shape=(32, 32, 1))

    X11 = Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), padding='same')(AA)
    X11 = BatchNormalization()(X11)
    X11 = tf.keras.layers.ReLU()(X11)  # Lambda(my_Relu6, name='relu6_2')(CC)

    sub_model = model_blocks(inputs=(16, 16, 8), iter_stage_2=1, iter_stage_3=1)
    X2 = sub_model(X11)
    X2 = GlobalAveragePooling2D()(X2)  # with GAP: 8970 parameters
    # X2=Flatten()(X2)  #with flatten 9930 parameters
    # X2=Dropout(0.2)(X2)

    X2 = Dense(units=10)(X2)  # activation='softmax',bias_initializer=bias

    my_model_raw = Model(inputs=AA, outputs=X2)
    #    my_model_raw.summary()
    #    plot_model(my_model_raw, show_shapes=True,expand_nested=False )
    return my_model_raw


my_model_raw = final_model()
my_model_raw.summary()


def count_trainable_variables(model):
    A = 0
    for i in range(len(model.trainable_variables)):
        A += np.prod((model.trainable_variables[i].shape))
    #  A+=np.prod((my_model.get_weights()[i].shape))
    print('number of trainable parameters in raw model={}'.format(A))


count_trainable_variables(my_model_raw)  # 8970