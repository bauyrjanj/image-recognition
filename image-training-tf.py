import numpy as np
import os
import cv2
import random
import tensorflow as tf
import time
import math

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
X = X/255
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


# Confirm that TensorFlow can access the GPU
device_name = tf.test.gpu_device_name()
if not device_name:
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Network architecture

# Images are stored in one-dimensional arrays of this length
img_size_flat = img_size*img_size

# Height and width of each image in Tuple
img_shape = (img_size, img_size)

# Number of color channels for the image
num_channels = 1

# Number of classes - two classes, positive or negative
num_classes = 2

# Convolutional layer 1
filter1_size = 5
num_filters1 = 16

# Convolutional layer 2
filter2_size = 5
num_filters2 = 16

# Convolutional layer 3
filter3_size = 5
num_filters3 = 16

# Pooling
window_size = 2
window_stride = 2

# Fully-connected dense 1
fc1_size=1024

# Fully-connected dense 2
fc2_size=2048

# Convolution stride
conv_stride=2


# Split the dataset into 70% vs 30% train and test sets
border = int(0.7*X.shape[0])
X_train = X[:border]
X_test = X[border:]
y_train = y[:border]
y_test = y[border:]


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

def weights(shape):
    weights = tf.Variable(tf.random.normal(shape=shape, stddev=0.05))
    return weights

# Conv 1
conv1_img_size = math.ceil(img_size/conv_stride)
shape_1 = [filter1_size, filter1_size, num_channels, num_filters1]
shape_bias1 = [num_channels, conv1_img_size, conv1_img_size, num_filters1]
conv1_weights = weights(shape_1)
bias_1 = tf.Variable(tf.ones(shape=shape_bias1))

def ConvNet1(image):
    # Conv1 layer
    conv1 = tf.nn.conv2d(input=image, filters=conv1_weights, strides=conv_stride, padding='SAME') # conv layer
    conv1+=bias_1
    conv1 = tf.nn.relu(conv1)                                                                     # relu layer
    conv1 = tf.nn.max_pool(input=conv1, ksize=window_size, strides=window_stride, padding='SAME') # pooling layer
    return conv1

conv1 = ConvNet1(X_train)
print(conv1.shape)

# Conv 2
conv2_img_size = math.ceil(conv1.get_shape()[1]/conv_stride)
shape_2 = [filter2_size, filter2_size, num_filters1, num_filters2]
shape_bias2 = [num_channels, conv2_img_size, conv2_img_size, num_filters2]
conv2_weights = weights(shape_2)
bias_2 = tf.Variable(tf.ones(shape=shape_bias2))

def ConvNet2(conv1):
    # Conv2 Layer
    conv2 = tf.nn.conv2d(input=conv1, filters=conv2_weights, strides=conv_stride, padding='SAME')
    conv2+=bias_2
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(input=conv2, ksize=window_size, strides=window_stride, padding='SAME')
    return conv2

conv2 = ConvNet2(conv1)
print(conv2.shape)

# Conv 3
conv3_img_size = math.ceil(conv2.get_shape()[1]/conv_stride)
shape_3 = [filter3_size, filter3_size, num_filters2, num_filters3]
shape_bias3 = [num_channels, conv3_img_size, conv3_img_size, num_filters3]
conv3_weights = weights(shape_3)
bias_3 = tf.Variable(tf.ones(shape=shape_bias3))

def ConvNet3(conv2):
    # Conv2 Layer
    conv3 = tf.nn.conv2d(input=conv2, filters=conv3_weights, strides=conv_stride, padding='SAME')
    conv3+=bias_3
    conv3 = tf.nn.relu(conv3)
    conv3 = tf.nn.max_pool(input=conv3, ksize=window_size, strides=window_stride, padding='SAME')
    return conv3

conv3 = ConvNet3(conv2)
print(conv3.shape)


# flatten
def flatten_layer(conv3):
    layer_shape = conv3.get_shape()
    num_features = layer_shape[1:4].num_elements()
    flat_layer = tf.reshape(conv3, [-1, num_features])
    return flat_layer, num_features

flat, features = flatten_layer(conv3)
print(features, flat.shape)

# Fully connected dense 1
num_features=features
fc1_shape = [num_features, fc1_size]
fc1_weights = tf.Variable(tf.random.normal(shape=fc1_shape))
bias_fc1 = tf.Variable(tf.ones([fc1_size]))

# Fully connected dense 2

num_features=features
fc2_shape = [fc1_size, fc2_size]
fc2_weights = tf.Variable(tf.random.normal(shape=fc2_shape))
bias_fc2 = tf.Variable(tf.ones([fc2_size]))


# Output layer
shape_out = [fc2_size, num_classes]
w_out = tf.Variable(tf.random.normal(shape=shape_out))
b_out = tf.Variable(tf.ones([num_classes]))


def make_prediction(flat_layer, fc1_weights, bias_fc1, fc2_weights, bias_fc2, w_out, b_out):
    # Fully connected dense 1
    fc1_product = tf.matmul(flat_layer, fc1_weights)
    fc1 = tf.keras.activations.relu(fc1_product + bias_fc1)
    fc1_drop = tf.nn.dropout(fc1, rate=0.5, seed=1)  # Drop 70% of the input elements randomly

    # Fully connected dense 2
    fc2_product = tf.matmul(fc1_drop, fc2_weights)
    fc2 = tf.keras.activations.relu(fc2_product + bias_fc2)
    fc2_drop = tf.nn.dropout(fc2, rate=0.5, seed=1)  # Drop 70% of the input elements randomly

    # Output layer
    output = tf.matmul(fc2_drop, w_out)
    prediction = tf.keras.activations.softmax(output + b_out)

    return prediction


def model(image):
    conv1 = ConvNet1(image)
    conv2 = ConvNet2(conv1)
    conv3 = ConvNet3(conv2)
    flat_layer, num_features = flatten_layer(conv3)

    predictions = make_prediction(flat_layer, fc1_weights, bias_fc1, fc2_weights, bias_fc2, w_out, b_out)
    return predictions

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Instantiate a loss function.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the metrics.
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

# Package the data into batches
batch_size=32
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(X_train.shape[0]).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).shuffle(X_test.shape[0]).batch(batch_size)

training_loss_values = []
training_accuracy_values = []
test_accuracy_values = []
train_steps = int(X_train.shape[0]/batch_size)
test_steps = int(X_test.shape[0]/batch_size)

# Training
start = time.time()
epochs = 100
for epoch in range(epochs):
    print("\nEpoch: {}/{}".format(epoch + 1, epochs))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_ds):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = model(x_batch_train)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch_train, logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, [conv1_weights, bias_1, conv2_weights, bias_2, conv3_weights, bias_3, fc1_weights, bias_fc1, fc2_weights, bias_fc2, w_out, b_out])

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, [conv1_weights, bias_1, conv2_weights, bias_2, conv3_weights, bias_3, fc1_weights, bias_fc1, fc2_weights, bias_fc2, w_out, b_out]))

        # Update training metric.
        train_acc_metric.update_state(y_batch_train, logits)

        # Log every 1874 batches.
        if step % train_steps == 0 and step != 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %s samples" % ((step + 1) * train_steps))

        # Display metrics at the end of every 1874 batch
        if step % train_steps == 0 and step != 0:
            train_acc = train_acc_metric.result()
            print("Training acc (for one batch): %.4f" % (float(train_acc),))

    # Run a test loop at the end of each epoch.
    for step, (x_batch_test, y_batch_test) in enumerate(test_ds):
        predictions = model(x_batch_test)
        # Update test metrics
        test_acc_metric.update_state(y_batch_test, predictions)
        if step % test_steps == 0 and step != 0:
            test_acc = test_acc_metric.result()
            print("Test acc: %.4f" % (float(test_acc),))

    training_loss_values.append(float(loss_value))
    training_accuracy_values.append(float(train_acc_metric.result()))
    test_accuracy_values.append(float(test_acc_metric.result()))
    # Reset metrics at the end of each epoch
    train_acc_metric.reset_states()
    test_acc_metric.reset_states()

print("Time taken: %.2fs" % (time.time() - start))






