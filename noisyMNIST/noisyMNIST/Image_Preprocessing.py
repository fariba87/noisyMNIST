


# handcrafted image processing

import numpy as np
import cv2

def preprocessing(img):
    # a kernel to sharp images (with enhancement)
    # reference: Book: "OpenCV 3 Computer Vision with Python Cookbook" By Alexey Spizhevoy
    kernel_sharpen = np.array([[-1,-1,-1,-1,-1],
                                 [-1,2,2,2,-1],
                                 [-1,2,8,2,-1],
                                 [-1,2,2,2,-1],
                                 [-1,-1,-1,-1,-1]]) / 8.0

    image1=img
    image2 = cv2.filter2D(image1, -1, kernel_sharpen)
    #median to remove noise
    image3 = cv2.medianBlur(image2, 3)
    ksize, sigma_color, sigma_space= 5, 5, 7
    #bilateral filter is better that gaussian in smoothing while saving details
    image4= cv2.bilateralFilter(image3, ksize, sigma_color, sigma_space)
    # resize to (32,32) 
    # Seam Carving can also be used for resizing
    # linear intepolation as we are enlarging and it is lighter than cubic
    image5=cv2.resize(image4, (32,32), interpolation = cv2.INTER_LINEAR)

    return image5


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# data have been been balanced by imblearn library (1000 image per class)



################################################################################
# I try erode, dilate, thresholding, contours and sorting based on , equalizehist

# how about raw data (just resized appropriately) as input?! to evaluate the power of network?!
# I should not try totally clean data, as the target is Generalization!

# try denoising AE!
# get a model that is trained by random-normal-noisy MNIST dataset(train, save,load) and then use it for prediction
# predictions can be thought as the preprocessed inputs
# but this approach is wrong since noise distribution is different  
##############################################################################################################
'''
    #plt.imshow(output_3,  cmap='gray')
    #otsu_thr, otsu_mask = cv2.threshold(output_3.copy(), 20, 255, cv2.THRESH_BINARY)
    #otsu_thr, otsu_mask = cv2.threshold(output_3.copy(), -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    plt.imshow(otsu_mask, cmap='gray')
    kernel=np.ones((1,1), np.uint8)
    otsu_mask=cv2.resize(otsu_mask, (20,32))
    otsu_mask=cv2.erode(otsu_mask, kernel, iterations=1)
    image2=cv2.dilate(otsu_mask, kernel, iterations=1)
    plt.imshow(image2, cmap='gray')
    
    contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_external = np.zeros(image.shape, image.dtype)
    for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(image_external, contours, i, 255, -1)
        
plt.figure(figsize=(10,3))
plt.subplot(121)
plt.axis('off')
plt.title('original')
plt.imshow(image, cmap='gray')
plt.subplot(122)
plt.axis('off')
plt.title('external')
plt.imshow(image_external, cmap='gray')

for contour in contours:
    area = cv2.contourArea(contour)
    print(area)

###########################################################################################
# failed!!

# try denoising AE
# get a model that is trained by random-normal-noisy MNIST dataset(train, save,load) and then use it for prediction
# predictions can be thought as the preprocessed inputs
# but this approach is wrong since noise distribution is different    
#cv2.imwrite('pic.jpg', x_train_preprocess[190])
   from tensorflow.keras.models import load_model 
model_de=load_model('C:/Users/scc/Desktop/DeepTask-Gata/denoise.hdf5')
AA=cv2.bitwise_not(x_train_preprocess[3000])
AA=cv2.resize(AA, (28,28))
AA.reshape((28,28,1))
AAA=np.expand_dims(AA, axis=0)
SS=model_de.predict(AAA)
plt.imshow(SS[0,:,:,0], cmap='gray')'''
