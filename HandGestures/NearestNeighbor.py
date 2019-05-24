import tensorflow as tf
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

##cap = cv2.VideoCapture(1)
##ret, frame = cap.read()
##
images = []
##
##while True:
##    ret, frame = cap.read()
##
##    cv2.imshow('frame', frame)
##    
##    if cv2.waitKey(1) & 0xFF == ord('q'):
##        break
##    
##    print('Say "Cheese"! Took a picture!')
##    time.sleep(3)
##    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
##    resized = cv2.resize(gray, (28, 28))
##    reshaped = np.reshape(resized, (784,))
##    images.append(reshaped)
##
images = np.asarray(images)

train = pd.read_csv('./sign_mnist_train/sign_mnist_train.csv')
test = pd.read_csv('./sign_mnist_test/sign_mnist_test.csv')

train_images = []
train_labels = []

test_images = []
test_labels = []

for rows in range(27455):
    image = train.values[rows][1 : ]
    train_images.append(image)
    one_hot_label = [0 for i in range(25)]
    one_hot_label[train.values[rows][0]] = 1
    train_labels.append(one_hot_label)

for rows in range(200):
    image = test.values[rows][1 : ]
    test_images.append(image)
    one_hot_label = [0 for i in range(25)]
    one_hot_label[test.values[rows][0]] = 1
    test_labels.append(one_hot_label)

test_images = np.asarray(test_images)

for classes in range(25):
    label = [0 for i in range(25)]
    label[classes] = 1
    try:
        index = train_labels.index(label)
    except:
        pass
    image = train_images[index]
    image = image.reshape((28, 28))
##    plt.subplot(6, 4, classes + 1)
##    plt.title('{}'.format(classes))
##    plt.imshow(image)
##plt.show()

x = tf.placeholder(tf.float32, shape = [None, 784])
centroid = tf.placeholder(tf.float32, 784)
distance = tf.reduce_sum(tf.abs(tf.add(x, tf.negative(centroid))),
                         reduction_indices = 1)
prediction = tf.arg_min(distance, 0)

##_ = tf.Variable(initial_value = 'fake_variable')

accuracy = 0

init = tf.global_variables_initializer()

##saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for var in range(len(test_images)):
##        print(var, test_images[var, :])
        nearest_neighbor = sess.run(prediction,
                                    feed_dict = {x : train_images,
                                    centroid : test_images[var]})
        print(var,
              'Prediction : ',
              np.argmax(train_labels[nearest_neighbor]),
              'Actual : ',
              np.argmax(test_labels[var]))
        if (np.argmax(train_labels[nearest_neighbor]) ==
                    np.argmax(test_labels[var])):
            accuracy = accuracy + 1/len(test_labels)

##    saver.save(sess, 'neighbor_model.ckpt')
    print(accuracy * 100, '%')






    
