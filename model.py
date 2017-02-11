### Import necessary packages
import numpy as np
import pandas as pd
import cv2
from os.path import basename
import json
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

### Convolution of a 1D signal with a gaussian filter of a given window length
def smooth(curve, window_len):
    
    # Gaussian parameters
    sigma = 1.
    mu = 0.
    # Setting up 'x', guarantees the gaussian window 'w' is centered and
    # independs on the value of mu and sigma
    x = np.arange(mu - 3*sigma, mu + 3.1*sigma, sigma/(window_len/6.))
    w = (np.exp((-1.*(x - mu)**2)/(2*sigma**2)))*(1./(np.sqrt(2*np.pi*sigma**2))) 

    y = np.convolve(w/w.sum(), curve, mode='same')
    return y

### Read the dataset and apply the 'smooth' function on its steering data
def smooth_dataset(file_name):
    dataset = pd.read_csv(file_name, header=0)
    dataset['steering'] = smooth(dataset['steering'][:], window_len=20)
    return dataset

### Load all the collected data
print("Loading collected data....")
c_rev = smooth_dataset('track1/full_center_rev1/driving_log.csv')
c = smooth_dataset('track1/full_center1/driving_log.csv')
tL_rev = smooth_dataset('track1/full_toLeft_rev1/driving_log.csv')
tL = smooth_dataset('track1/full_toLeft1/driving_log.csv')
tR_rev = smooth_dataset('track1/full_toRight_rev1/driving_log.csv')
tR = smooth_dataset('track1/full_toRight1/driving_log.csv')
c2 = smooth_dataset('track1/full_center2/driving_log.csv')
c_rev2 = smooth_dataset('track1/full_center_rev2/driving_log.csv')
tL2 = smooth_dataset('track1/full_toLeft2/driving_log.csv')
tL_rev2 = smooth_dataset('track1/full_toLeft_rev2/driving_log.csv')
tR2 = smooth_dataset('track1/full_toRight2/driving_log.csv')
tR_rev2 = smooth_dataset('track1/full_toRight_rev2/driving_log.csv')
print("Collected data loaded.")

### Read the original image, crop and resize it to a 32x32 format
def read_images(dataset, bias=0):
    img_data = np.empty((dataset.shape[0], 32, 32, 3))
    for i in range(dataset.shape[0]):
        img = cv2.imread('IMG/' + basename(dataset[i]))
        img = cv2.resize(img[80:120, 0:320, :],
                         None, fx=32./320., fy=32./40.,
                         interpolation=cv2.INTER_AREA)
        img_data[i, :, :, :] = img
    return img_data

### Load images and corresponding steering angles from given dataframe
def load_dataset_from_log(driving_log_df):
    center_img, center_steering = load_col_from_log(driving_log_df, col_name='center')
    left_img, left_steering = load_col_from_log(driving_log_df, col_name='left', bias=0.15)
    right_img, right_steering = load_col_from_log(driving_log_df, col_name='right', bias=-0.15)
    img_data = np.r_[center_img, left_img, right_img]
    steering_data = np.r_[center_steering,
                          left_steering,
                          right_steering]
    return img_data, steering_data

### Load images from a single camera and their corresponding steering
### angles with the appropriate bias from given dataframe
def load_col_from_log(driving_log_df, col_name, bias=0.):
    img_data = read_images(driving_log_df[col_name], bias=bias)
    steering_data = driving_log_df['steering'] + bias
    return img_data, steering_data

### Preprocess all the collected data
print("Preprocessing data....")
c_img, c_steering = load_dataset_from_log(c)
c_rev_img, c_rev_steering = load_dataset_from_log(c_rev)

tL_img, tL_steering = load_col_from_log(tL, col_name='center', bias=-0.2)
tL_rev_img, tL_rev_steering = load_col_from_log(tL_rev, col_name='center', bias=-0.2)

tR_img, tR_steering = load_col_from_log(tR, col_name='center', bias=0.2)
tR_rev_img, tR_rev_steering = load_col_from_log(tR_rev, col_name='center', bias=0.2)

c2_img, c2_steering = load_dataset_from_log(c2)
c_rev2_img, c_rev2_steering = load_dataset_from_log(c_rev2)

tL2_img, tL2_steering = load_col_from_log(tL2, col_name='center', bias=-0.2)
tL_rev2_img, tL_rev2_steering = load_col_from_log(tL_rev2, col_name='center', bias=-0.2)

tR2_img, tR2_steering = load_col_from_log(tR2, col_name='center', bias=0.2)
tR_rev2_img, tR_rev2_steering = load_col_from_log(tR_rev2, col_name='center', bias=0.2)

img_train = np.r_[c_img, c_rev_img,
                         tL_img, tL_rev_img,
                         tR_img, tR_rev_img,
                         c2_img, c_rev2_img,
                         tL2_img, tL_rev2_img,
                         tR2_img, tR_rev2_img]
steering_train = np.r_[c_steering, c_rev_steering,
                             tL_steering, tL_rev_steering,
                             tR_steering, tR_rev_steering,
                             c2_steering, c_rev2_steering,
                             tL2_steering, tL_rev2_steering,
                             tR2_steering, tR_rev2_steering]

### Split the training data in 3 subsets
### to ballance the cases during training
img_z = img_train[((steering_train <= 0.1) & (steering_train >= -0.1)), :, :, :]
img_p = img_train[steering_train > 0.1, :, :, :]
img_n = img_train[steering_train < -0.1, :, :, :]
str_z = steering_train[((steering_train <= 0.1) & (steering_train >= -0.1))]
str_p = steering_train[steering_train > 0.1]
str_n = steering_train[steering_train < -0.1]
print("Data preprocessed.")

### DNN architacture used to learn from data
### how to control the simulator
model = Sequential()
model.add(Convolution2D(4, 1, 1, border_mode='valid', input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(Convolution2D(8, 5, 5, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(8, 1, 1, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(16, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(16, 1, 1, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(1))

### Data augmentation strategies
def preprocess_img(img, steering):
    
    # 50% chance of flipping the image to augment
    # positive data points with negative ones and
    # vice versa
    flip = np.random.uniform(low=-0.5, high=0.5)
    if flip >= 0:
        img = cv2.flip(img, 1)
        steering = -steering
    
    # Gamma correction to augment data with respect
    # to brightness
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV)
    img = img/255.
    gamma = np.random.uniform(low=0.01, high=2.0)
    img[:, :, 2] = img[:, :, 2]**gamma
    img = cv2.cvtColor((img*255.).astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # Image normalization - guarantees the data
    # has zero mean and std less than 1
    img = img - img.mean()
    img = img/img.max()
    
    # Added gaussian noise to augment data with
    # respect to loss of information
    sigma = np.random.uniform(low=0.001, high=0.01)
    noise = np.random.randn(32, 32, 3)*sigma
    img = img + noise
    return img, steering

### Generator used to augment and ballance the training data
def batch_generator(img0_dataset, steering0_dataset, 
                    imgP_dataset, steeringP_dataset,
                    imgN_dataset, steeringN_dataset,
                    batch_size=1800):
    
    img_batch = np.empty((batch_size, 32, 32, 3))
    steering_batch = np.empty((batch_size,))
    
    while True:
        
        batch_slice = int(batch_size/3)
        
        # Data close to zero
        img0_dataset, steering0_dataset = shuffle(img0_dataset, 
                                                  steering0_dataset)
        for i in range(batch_slice):
            img_batch[i, :, :, :], steering_batch[i] = preprocess_img(img0_dataset[i, :, :, :], steering0_dataset[i])
        
        # Positive data
        imgP_dataset, steeringP_dataset = shuffle(imgP_dataset, 
                                                  steeringP_dataset)
        for i in range(batch_slice, 2*batch_slice):
            i_ = i - batch_slice
            img_batch[i, :, :, :], steering_batch[i] = preprocess_img(imgP_dataset[i_, :, :, :], steeringP_dataset[i_])
        
        # Negative data
        imgN_dataset, steeringN_dataset = shuffle(imgN_dataset, 
                                                  steeringN_dataset)
        for i in range(2*batch_slice, batch_size):
            i_ = i - 2*batch_slice
            img_batch[i, :, :, :], steering_batch[i] = preprocess_img(imgN_dataset[i_, :, :, :], steeringN_dataset[i_])
        
        yield (img_batch, steering_batch)

### Train model to minimize 'mse' loss function
### with default Adam optimizer
print("Training model....")
model.compile(loss='mean_squared_error', optimizer='adam')
datagen = batch_generator(img_z, str_z, 
                          img_p, str_p,
                          img_n, str_n)
history = model.fit_generator(datagen, 
                              samples_per_epoch=48600, 
                              nb_epoch=10, 
                              verbose=1)
print("Model trained.")

### Save model parameters
print("Saving model parameters....")
model.save_weights('model.h5')
model_json = model.to_json()
with open('model.json', 'w') as f:
    json.dump(model_json, f)
print("Parameters saved.")