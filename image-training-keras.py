import numpy as np
import os
import cv2
import random
import time
import math
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras import metrics

datadir = "data"
directory = ['negative', 'positive']

# Create your training data set
training_data=[]
img_size=600

def create_training_data():
    for folder in directory:
        path = os.path.join(datadir, folder)
        label = directory.index(folder)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, label])
            except Exception as e:
                pass

create_training_data()

# Shuffle them to get a good mix of different labels
random.shuffle(training_data)

# Seperate features and labels into separate arrays
X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X)
X = X/255.0
y = np.array(y)

# reshape as required by Tensorlow/Keras
X = X.reshape(-1,img_size, img_size, 1)
y = np.array(y).reshape(-1,1)

# Check data size and types
print(X.shape)
print(y.shape)
print(X.dtype)
print(y.dtype)

X = X.astype('float32')
print(X.dtype)

# Split the dataset into 70% vs 30% train and test sets
border = int(0.7*X.shape[0])
X_train = X[:border]
X_test = X[border:]
y_train = y[:border]
y_test = y[border:]


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Number of color channels for the image
num_channels = 1

# Height and width of each image in Tuple
img_shape = (img_size, img_size, num_channels)

# Number of classes - two classes, positive or negative
num_classes = 2

# Convolutional layer 1
filter1_size = 3
num_filters1 = 64

# Convolutional layer 2
filter2_size = 5
num_filters2 = 128

# Convolutional layer 3
filter3_size = 5
num_filters3 = 256

# Pooling
window_size = 3
window_stride = 1

# Fully-connected dense 1
fc1_size=2048

# Fully-connected dense 2
fc2_size=1024

# Convolution stride
conv_stride=3

# batch size
batch_size = 32

# Confirm that TensorFlow can access the GPU
device_name = tf.test.gpu_device_name()
if not device_name:
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

model = Sequential([
    Conv2D(filters = num_filters1, kernel_size=filter1_size, strides=(conv_stride,conv_stride), padding='valid', activation='relu', input_shape=img_shape),
    MaxPooling2D(pool_size=(window_size, window_size), strides=window_stride),
    Conv2D(filters= num_filters2, kernel_size=filter2_size, strides=(conv_stride, conv_stride), padding='valid', activation='relu'),
    MaxPooling2D(pool_size=(window_size, window_size), strides=window_stride),
    Conv2D(filters=filter3_size, kernel_size=filter3_size, strides=(conv_stride, conv_stride), padding='valid', activation='relu'),
    MaxPooling2D(pool_size=(window_size, window_size), strides=window_stride),
    Flatten(),
    Dense(fc1_size, activation='relu'),
    Dropout(0.5),
    Dense(fc2_size, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')

])

model.compile(optimizer = optimizers.RMSprop(learning_rate=1e-5), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(X_train, y_train, batch_size=batch_size, epochs = 100, validation_split = 0.3, callbacks = tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=10))

model.save('cnn_ben.h5')


