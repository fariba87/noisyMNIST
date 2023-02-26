import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import cv2
from tensorflow.keras.utils import to_categorical

np.random.seed(11)
tf.random.set_seed(11)
from augment_class_wise import Images, Labels
import imblearn
from imblearn.over_sampling import SMOTE
import numpy as np
from Image_Preprocessing import preprocessing

absolute_path = os.path.dirname(__file__)
absolute_path

# %%####################################################################################
def create_subdirectory():
    import os
    os.mkdir('Train_class')
    os.mkdir('Val_class')
    os.chdir('Train_class')
    os.getcwd()
    import shutil
    for category in range(10):
        os.mkdir(str(category))
    os.chdir('D:/GATA-Fariba rezaei/Data')

    dir_train_class = 'C:/Users/scc/Desktop/GATA/Train_class' # i had split the data - but not attached to rar file
    for category in range(10):
        for i, file_path in enumerate(Train_sub):
            label, _ = train_list[i].split('.')[0].split('_')
            if np.uint32(label) == category:
                shutil.copy(Train_sub[i], os.path.join(dir_train_class, str(category)))
    os.chdir('C:/Users/scc/Desktop/DeepTask-Gata/Data')

    Val_sub = [os.path.join(val_dir, val_list[i]) for i in range(len(val_list))]
    dir_val_class = 'C:/Users/scc/Desktop/GATA/Val_class'
    os.chdir(dir_val_class)

    for category in range(10):
        os.mkdir(str(category))
        for i, file_path in enumerate(Val_sub):
            label, _ = val_list[i].split('.')[0].split('_')
            if np.uint32(label) == category:
                shutil.copy(Val_sub[i], os.path.join(dir_val_class, str(category)))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# data directory
#data_dir = 'D:/GATA-Fariba rezaei/Data'
data_dir = os.chdir("..")
data_dir = os.chdir("..")
data_dir=os.path.abspath(os.curdir)
data_dir= data_dir+'/Data' 
train = 'train'
val = 'valid'
train_dir = os.path.join(data_dir, train)
val_dir = os.path.join(data_dir, val)
train_list = os.listdir(train_dir)
val_list = os.listdir(val_dir)
# glob.glob('train/*.jpg'')
import imgaug.augmenters as iaa

'''augmentation=iaa.Sequential([
    iaa.Rotate(-30,30),
    iaa.Affine(translate_percent={"x":(-0.2,0.2), "y":(-0.2,0.2)}, scale=(0.2,0.3)}),
    iaa.LinearContrast()
    ])
augmened_images=augmentation(images=images)'''


#####################################################################################

# split image and label and apply for train and val folder


def get_image_and_label_by_path(data_dir, data_list):
    # height of images:
    # np.unique([cv2.imread(os.path.join(train_dir,train_list[i])).shape[0] for i in range(len(train_list))])    #32
    # np.unique([cv2.imread(os.path.join(train_dir,train_list[i])).shape[1] for i in range(len(train_list))])     #array([ 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])

    '''
    Inputs: path to train or valid
    Output: Images:list of numpy array  (still different sizes)
            label :np array  {0,1,...}

    '''
    im_label = []
    Image = []
    for i, _ in enumerate(data_list):
        #  Img=cv2.resize(cv2.imread(os.path.join(data_dir,data_list[i]), 0),(20,32)) #wait for resize!!!
        Img = cv2.imread(os.path.join(data_dir, data_list[i]), 0)
        Image.append(Img)
        label = data_list[i].split('.')[0].split('_')[0]
        lab = np.uint32(label)
        im_label.append(lab)
    return Image, np.array(im_label),  # np.array(Image_)


x_train, y_train = get_image_and_label_by_path(train_dir, train_list)
x_val, y_val = get_image_and_label_by_path(val_dir, val_list)

#####################################################################################
# exploring dataset

num_classes = len(np.unique(y_train))
print(f' the number of train data:   # {len(y_train)} ')
print(f' the number of validation data:    # {len(y_val)} ')
print(f' the number of classes:    # {num_classes} ')

plt.figure(figsize=(10, 10))
plt.title('distribution of classes in train data')
sns.countplot(x=y_train)

plt.figure(figsize=(10, 10))
plt.title('distribution of classes in  validation data')
sns.countplot(x=y_val)
# plt.savefig(args, kwargs)
from sklearn.utils import class_weight

class_weight = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
print('class weight:\n', class_weight)
# Data Augmentation is not as good as SMOTE to solve imbalanced class problem in images.


#####################################################################################

#                               to see how width and height is varying among data
# np.unique([cv2.imread(os.path.join(train_dir,train_list[i])).shape[0] for i in range(len(train_list))])    #32
# np.unique([cv2.imread(os.path.join(train_dir,train_list[i])).shape[1] for i in range(len(train_list))])     #array([ 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])

A = [cv2.imread(os.path.join(train_dir, train_list[i])).shape[1] for i in range(len(train_list))]
np.argmin(A)  # image 5800 shape:(_,9)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def image_label_data(only_resize=True,
                     only_resize_and_balance=True,
                     preprocess_and_resize=False,
                     preprocess_and_resize_and_balanced=False,
                     augment_class_wise_=False,
                     augment_class_wise_preprocess=False):
    '''
    function to return x_train , y_train , x_val , y_val (len ,32,32,1)
    only_resize=True,                  : just resize all images to (32,32,1)
    only_resize_and_balance=True,      : first resize and then run SMOTE algorithm to balance the data
    preprocess_and_resize=True,        : first preprocess images (image processing) and then resize
    preprocess_and_resize_and_balanced=True   : first preprocess, resize, SMOTE


    # Synthetic Minority Reconstruction Technique (SMRT) is based on VAE and SMOTE to generate wise image samples(in comparison to
    #SMOTE which is not so good for images ! but i couldnt install it!)
    # i didnt apply SMRT !
    '''

    if only_resize:
        def resizing(img):
            image = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)
            return image

        import numpy as np
        x_train1 = np.array(list(map(resizing, x_train)))
        x_train1 = np.expand_dims(x_train1, -1)

        y_train1 = to_categorical(y_train, 10)

        x_val1 = np.array(list(map(resizing, x_val)))
        x_test = np.expand_dims(x_val1, -1)

        y_test = to_categorical(y_val, 10)

        return x_train1, y_train1, x_test, y_test
    elif only_resize_and_balance:
        def resizing(img):
            image5 = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)
            return image5

        x_train1 = np.array(list(map(resizing, x_train)))
        yraw = to_categorical(y_train1, 10)
        oversample = SMOTE()
        Xnew, ynew = oversample.fit_resample(x_train1.reshape((7623, -1)), yraw)
        #  Xnew=np.load('Xnew.npy') # khodesh resize shode
        x_train_balanced = np.reshape(Xnew, (10000, 32, 32))
        x_train_balanced = np.expand_dims(x_train_balanced, -1)

        # Ynew=np.load('Ynew.npy') #khodesh categorical hast
        # y_train_encode=to_categorical(Yne#)
        #  y_train_balanced=Ynew
        x_raw_val = np.array(list(map(resizing, x_val)))
        x_raw_val = np.expand_dims(x_raw_val, -1)

        y_raw_val = to_categorical(y_val, 10)
        return x_train_balanced, ynew, x_raw_val, y_raw_val

    elif preprocess_and_resize:
        import numpy as np
        # preprocess input images based on our processing func
        x_train_preprocess = np.array(list(map(preprocessing, x_train)))

        x_train_preprocess = np.expand_dims(x_train_preprocess, -1)
        x_val_preprocess = np.array(list(map(preprocessing, x_val)))
        x_val_preprocess = np.expand_dims(x_val_preprocess, -1)
        # one hot encoding labels
        y_train_encode = to_categorical(np.uint32(y_train), 10)
        y_val_encode = to_categorical(np.uint32(y_val), 10)

        return x_train_preprocess, y_train_encode, x_val_preprocess, y_val_encode


    elif preprocess_and_resize_and_balanced:

        import imblearn
        from imblearn.over_sampling import SMOTE
        import numpy as np
        x_train1 = np.array(list(map(preprocessing, x_train)))
        oversample = SMOTE()

        Xnew, ynew = oversample.fit_resample(x_train1.reshape((7623, -1)), y_train)
        y_train1 = to_categorical(ynew, 10)

        x_train_balanced = np.reshape(Xnew, (10000, 32, 32))
        x_train_balanced = np.expand_dims(x_train_balanced, -1)
        x_val_preprocess = np.array(list(map(preprocessing, x_val)))
        x_val_preprocess = np.expand_dims(x_val_preprocess, -1)
        y_raw_val = to_categorical(y_val, 10)  # y_train_encode=to_categorical(Ynew) #khodesh nist?
        return x_train_balanced, y_train1, x_val_preprocess, y_raw_val
    elif augment_class_wise_:
        def resizing(img):
            image5 = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)
            return image5

        import numpy as np
        x_raw_val = np.array(list(map(resizing, x_val)))
        x_val_ = np.expand_dims(x_raw_val, -1)
        y_val_ = to_categorical(y_val, 10)
        Images_train = np.expand_dims(Images, -1)
        Labels_train = to_categorical(Labels, 10)
        return Images_train, Labels_train, x_val_, y_val_
    elif augment_class_wise_preprocess:

        import numpy as np
        x_train_preprocess = np.array(list(map(preprocessing, Images)))

        x_train_preprocess = np.expand_dims(x_train_preprocess, -1)
        x_val_preprocess = np.array(list(map(preprocessing, x_val)))
        x_val_preprocess = np.expand_dims(x_val_preprocess, -1)
        # one hot encoding labels
        y_train_encode = to_categorical(np.uint32(Labels), 10)
        y_val_encode = to_categorical(np.uint32(y_val), 10)
        return x_train_preprocess, y_train_encode, x_val_preprocess, y_val_encode