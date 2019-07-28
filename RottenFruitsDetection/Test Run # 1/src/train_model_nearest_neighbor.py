import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cv2
import time
from imutils import paths

train_images = []
train_labels = []

test_images = []
test_labels = []

train_images = np.asarray(train_images)
test_images = np.asarray(test_images)

def path_images(folder_name, image_array, label_array):
    for image_path in paths.list_images(folder_name):
        
        image = image_path.split('/')[-1]

        image = cv2.imread(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (28, 28))

        image = np.reshape(image, (784,))

        image_array.append(image)

        if 'apple' in image_path:
            label_array.append([0, 1])
        elif 'rotten' in image_path:
            label_array.append([1, 0])
            
path_images('train', train_images, train_labels)
path_images('test', test_images, test_labels)

trainX = tf.placeholder(tf.float32, shape = [None, 784])
centroid = tf.placeholder(tf.float32, 784)
distance = tf.reduce_sum(tf.abs(tf.add(trainX, tf.negative(centroid))),
                         reduction_indices = 1)
prediction = tf.arg_min(distance, 0)

_ = tf.Variable(initial_value = 'fake_variable')

accuracy = 0

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for var in range(len(test_images)):
        nearest_neighbor = sess.run(prediction,
                                    feed_dict = {trainX : train_images,
                                    centroid : test_images[var]})
        print(var,
              'Prediction : ',
              np.argmax(train_labels[nearest_neighbor]),
              'Actual : ',
              np.argmax(test_labels[var]))
        if (np.argmax(train_labels[nearest_neighbor]) ==
                    np.argmax(test_labels[var])):
            accuracy = accuracy + 1/len(test_labels)

    saver.save(sess, 'neighbor_model.ckpt')
    print(accuracy * 100, '%')
