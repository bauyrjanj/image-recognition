import tensorflow as tf
from tensorflow import keras
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from sklearn.metrics import accuracy_score

model = keras.models.load_model('cnn_ben.h5')

print(model.summary())

datadir = 'didnt_pass'
directory = ["negative", "positive"]

# Create your training data set
testing_data=[]
img_size=600

def create_testing_data():
    for folder in directory:
        path = os.path.join(datadir, folder)
        label = directory.index(folder)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                testing_data.append([new_array, label])
            except Exception as e:
                pass

create_testing_data()


# Seperate features and labels into separate arrays
X = []
y = []
for features, label in testing_data:
    X.append(features)
    y.append(label)

X = np.array(X)
X = X/255.0
y = np.array(y)

# reshape as required by Tensorlow/Keras
X = X.reshape(-1,img_size, img_size, 1)

pred = model.predict(X)
print(pred)
print(pred.argmax(axis=1))
print("Accuracy: {}".format(sum(y==pred.argmax(axis=1))/len(y)))



