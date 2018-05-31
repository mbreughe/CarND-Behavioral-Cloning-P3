# Setup some libraries
import os
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from math import ceil
#keras packages
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Cropping2D
from keras.layers import Lambda
from keras.layers import Dropout

global correction
global use_cropping
global use_left_right
global start_learn_rate
use_left_right = False
use_cropping = False
correction = 1.0 # 2.0 --> worse than no data (2.6 mse); 4.0 pretty good (0.0052, but last epoch very bad); 6.0 --> 26 mse; 5 --> 0.008 (last epoch very bad); 3.0 --> very bad; 3.5 --> 8; 3.75 --> 9; 4.1 --> 11; 3.4 -> 7.7; 3.9 --> 0.025, last 2 epoch very bad; 3.8 --> 0.006 last 2 epoch 10 
start_learn_rate = 0.001
batch_size = 32
num_epochs = 4
test_size = 0.33

from collections import namedtuple

img_data = namedtuple("img_data", ["img_path", "flip"])

# Read the csv file and add files
# It also flips your images among vertical axis to create additional data
# use_left_right attempts to use left and right images in addition to center image.
# it does this by using the global "correction" factor
# repeat is a failed experiment where I copy some images multiple times
# --> this is bad as your validation set is likely to contain copies of your train data set!
def add_data(imgs, steering, data_path, use_left_right=False, repeat=1):
    print("Adding L/R data with correction {}".format(correction))
    csv_f = os.path.join(data_path, "driving_log.csv")
    
    fnames = []
    raw_steering = []
    with open(csv_f) as ifh:

        reader = csv.reader(ifh, delimiter=',')
        next(reader)
        for line in reader:
            rel_path = "/".join(line[0].split("\\")[-2:])
            steer = float(line[3])
            # Duplicate data by flipping
            fnames.append(os.path.join(data_path, rel_path))
            raw_steering.append(steer)
            
            if use_left_right:
                rel_path_left = "/".join(line[1].strip().split("\\")[-2:])
                rel_path_right = "/".join(line[2].strip().split("\\")[-2:])
                steer_left = steer + correction
                steer_right = steer - correction
                fnames.append(os.path.join(data_path, rel_path_left))
                fnames.append(os.path.join(data_path, rel_path_right))
                raw_steering.append(steer_left)
                raw_steering.append(steer_right)
    
    for fname, label in zip(fnames, raw_steering):
        # Repeat a number of times so we can add more weight to certain pictures
        for i in range(repeat):
            # Flip the image
            imgs.append(img_data(fname, False))
            steering.append(label)
            
            imgs.append(img_data(fname, True))
            steering.append(-label)

# Generator that does the actual loading of the image
from sklearn.utils import shuffle
def batchGenerator(features, labels, batch_size):
    features, labels = shuffle(features, labels)
    while True:
        for i in range(0, len(features), batch_size):
            X_data = list()
            y_data = list()
            for im, label in zip(features[i:i+batch_size], labels[i:i+batch_size]):
                cur_img = cv2.imread(im.img_path)
                if im.flip:
                    cur_img = np.fliplr(cur_img)
                X_data.append(cur_img)
                y_data.append(label)
            
            yield (np.array(X_data), np.array(y_data))

# Initial model
def createLenet():
    
    
    model = Sequential()
    # Crop off sky and hood of car
    #model.add(Cropping2D(cropping=((50,20), (1,1)), input_shape=(160, 320,3)))
    # first set of CONV => RELU => POOL
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320,3)) )
    model.add(Convolution2D(20, 5, 5, border_mode="same") )
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL
    model.add(Convolution2D(50, 5, 5, border_mode="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(1))

    return model

# Custom way of cropping
def rookieCrop(x):
    import tensorflow as tf
    mask = np.ones(x.get_shape()[1:])
    mask[:50,:,:] = 0
    mask[-20:,:,:] = 0
    cropper = tf.constant(mask, dtype=tf.float32)
    output = tf.multiply(x, cropper)
    #return output
    return x[:,50:-20,:,:]

def createNvidia():
    model = Sequential()
    
    # Use official cropping layer from keras --> buggy during inference!
    if use_cropping:
        model.add(Cropping2D(cropping=((50,20), (1,1)), input_shape=(160, 320,3)))
        model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    else:
        model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320,3)) )
        # Use custom made cropping layer
        model.add(Lambda(rookieCrop, output_shape=(90, 320,3)))
        # This was used before custom cropping layer added
        #model.add(MaxPooling2D(pool_size=(2,2)))   
    
    model.add(Convolution2D(24, 5, 5, subsample=(2,2)))
    
    model.add(Convolution2D(36, 5, 5, subsample=(2,2)))
    
    model.add(Convolution2D(48, 5, 5, subsample=(2,2)))
    
    model.add(Convolution2D(64, 3, 3))
    
    model.add(Convolution2D(64, 3, 3))
    
    model.add(Flatten())
    
    model.add(Dropout(p=0.5))
    
    model.add(Dense(100, activation="relu"))
    
    model.add(Dense(50, activation="relu"))
    
    model.add(Dense(10, activation="relu"))
    
    model.add(Dense(1))
    
    return model

#Questions: 
# simulator: recovery training really hard since the road edges are too high
# how is aceleration and braking set??
# recording takes too much time: could it be done without rendering?
def train(model, imgs, steering):
    print ("batch: {}, ecochs: {}, test_split: {}".format(batch_size, num_epochs, test_size))
    train_features, validation_features, train_labels, validation_labels = train_test_split(imgs, steering, test_size=test_size)
    train_data = batchGenerator(train_features, train_labels, batch_size)
    val_data = batchGenerator(validation_features, validation_labels, batch_size)

    model.fit_generator(train_data, len(train_features), num_epochs, validation_data = val_data, verbose = 1, nb_val_samples=len(validation_features))

# Experiment with left and right images
def left_right_test():
    use_left_right = True
    for i in np.arange(0.0, 2.0, 0.01):
        imgs = []
        steering = []
        global correction
        correction = i
        print ("Correction: {}".format(correction))
        add_data(imgs, steering, "data/course_train", use_left_right)
        #add_data(imgs, steering, "data/mb_train/clean_track1", use_left_right)
        #add_data(imgs, steering, "data/mb_train/clean_track1_clockwise", use_left_right)
        #add_data(imgs, steering, "data/mb_train/clean_track1_recov", use_left_right)
        #add_data(imgs, steering, "data/mb_train/track1_recov") # terrible
        
        train(model, imgs, steering)

# Model from scratch
#model = createNvidia()
#model.compile(optimizer='Adam', loss='mse')

# Best models so far
#model = keras.models.load_model("data/lowres_bs_512_udacity_data.h5")
#model = keras.models.load_model("data/lowres_bs_512_udacity_data_bridge_bend_bis_bis.h5")
model = keras.models.load_model("data/tip_top.h5")
model.optimizer = keras.optimizers.Adam(lr=start_learn_rate)
print(model.summary())


batch_size = 512
#left_right_test()
        
imgs = []
steering = []

add_data(imgs, steering, "data/course_train", use_left_right=False)
add_data(imgs, steering, "data/mb_train/issue_bridge", use_left_right=False)
add_data(imgs, steering, "data/mb_train/issue_bridge_bis", use_left_right=False)
add_data(imgs, steering, "data/mb_train/issue_bend", use_left_right=False)
add_data(imgs, steering, "data/mb_train/issue_bend_bis", use_left_right=False)
add_data(imgs, steering, "data/mb_train/issue_bend_2", use_left_right=False)
add_data(imgs, steering, "data/mb_train/issue_bend_2_bis", use_left_right=False)
add_data(imgs, steering, "data/mb_train/issue_bend_2_ter", use_left_right=False)
add_data(imgs, steering, "data/mb_train/issue_bend_4", use_left_right=False)
add_data(imgs, steering, "data/mb_train/issue_bend_2_quater", use_left_right=False)
add_data(imgs, steering, "data/mb_train/issue_bend_2_cinqo", use_left_right=False)
add_data(imgs, steering, "data/mb_train/issue_border", use_left_right=False)
add_data(imgs, steering, "data/mb_train/clean_track1_bis_p1", use_left_right=False)
#add_data(imgs, steering, "data/mb_train/clean_track1", use_left_right=False)
#add_data(imgs, steering, "data/mb_train/clean_track1_clockwise", use_left_right=False)
add_data(imgs, steering, "data/mb_train/clean_track1_recov", use_left_right=False)

train(model, imgs, steering)
model.save("data/tip_top_bend_4_recov.h5")

# Experiment with different batch sizes
#for bs in [32, 64, 128, 512]:
#    batch_size = bs
#    train(imgs, steering)

#best so far: dropout of 0.5 and udacity data only
