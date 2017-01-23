### Import necessary packages
import numpy as np
from os.path import isfile
import pickle
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import json

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

pickle_file = 'train_data.pickle'

if isfile(pickle_file):
    with open(pickle_file, mode='rb') as f:
        train = pickle.load(f)
    img_train = train['img']
    steering_train = train['steering']


print(img_train.shape, steering_train.shape)

img_train, steering_train = shuffle(img_train, steering_train)

### Normalize the image data
img_train = img_train/255. - 0.5


model = Sequential()
model.add(Convolution2D(4, 1, 1, border_mode='valid', input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(Convolution2D(8, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(16, 1, 1, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(16, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(4, 1, 1, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(16, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(img_train, steering_train, nb_epoch=20, validation_split=0.1)


model.save_weights('teste.h5')

data = model.to_json()
with open('teste.json', 'w') as f:
    json.dump(data, f)
