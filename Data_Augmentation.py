"""
Created on Wed Mar 29 11:56:00 2023

@author: Aleena

"""

from keras.preprocessing.image import ImageDataGenerator
from skimage import io

datagen = ImageDataGenerator(
        rotation_range=30,     #rotation between 0 and 45
        width_shift_range=0.2,   #% shift
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest', cval=125)    #try others like constant, reflect, wrap


######################Single Image####################################

# Loading a sample image  
x = io.imread('F:/Data_Augmentation/train/test1.jpeg') 
x = x.reshape((1, ) + x.shape)  #Array with shape (1, 256, 256, 3)

i = 0
for batch in datagen.flow(x, batch_size=16,  
                          save_to_dir='augmented', 
                          save_prefix='aug', 
                          save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely  


#######################Multiple Images##############################
#Read each image and supply an array to datagen via flow method

dataset = []

import numpy as np
from skimage import io
import os
from PIL import Image

image_directory = 'F:/data_augmentation/train/'
SIZE_X = 768
SIZE_Y = 1024
dataset = []

my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):
    if (image_name.split('.')[1] == 'jpeg'):
        image = io.imread(image_directory + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE_X,SIZE_Y))
        dataset.append(np.array(image))

x = np.array(dataset)

#I'm using 200 because I have 200 original images, change is as per your need
#provide a output folder like 'augmented'
#give a prefix to your new filenames
#pick a format for image like jpeg, jpg, png
   
i = 0
for batch in datagen.flow(x, batch_size=200,  
                          save_to_dir='F:/data_augmentation/augmented', 
                          save_prefix='aug', 
                          save_format='jpeg'): 
    i += 1
    if i > 2:
        break  
# this will run the entire thing three times, giving 200*3 i.e. 600 augmented images