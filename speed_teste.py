import numpy as np
from os.path import basename, isfile
import pickle
import time
from keras.models import model_from_json
import json
# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

pickle_file = 'train_data.pickle'

if isfile(pickle_file):
    print('File already exists.')
    with open(pickle_file, mode='rb') as f:
        train = pickle.load(f)
    img_train = train['img']
    steering_train = train['steering']

print('Dataset size:')
print(img_train.shape, steering_train.shape)

### Normalize the image data
img_train = img_train/255. - 0.5

print('Done normalizing.')

with open('teste.json', 'r') as jfile:
        model = model_from_json(json.loads(jfile.read()))

model.compile("adam", "mse")
weights_file = 'teste.h5'
model.load_weights(weights_file)

print('Done loading model.')

print(model.evaluate(img_train, steering_train))
print('Done evaluating model.')

print('Time for prediction:')

start_time = time.time()
model.predict(img_train[0:1, :, :, :])
print("--- %s seconds ---" % (time.time() - start_time))