import os
import cv2
import random
import numpy as np


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

random.shuffle(training_data)

X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X)
X = X/255.0
# reshape as required by Tensorlow/Keras
X = X.reshape(-1,img_size, img_size, 1)
# reshape as required by Tensorflow/Keras
y = np.array(y).reshape(-1,1)

# Save the data
import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()



