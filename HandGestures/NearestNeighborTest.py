import tensorflow as tf
import numpy as np
import cv2
import time
import pandas as pd

# opencv
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

fgbg = cv2.createBackgroundSubtractorMOG2()

# tensorflow
tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape = [None, 784])
centroid = tf.placeholder(tf.float32, 784)
distance = tf.reduce_sum(tf.abs(tf.add(x, tf.negative(centroid))),
                         reduction_indices = 1)
prediction = tf.arg_min(distance, 0)

_ = tf.Variable(initial_value = 'fake_variable')

accuracy = 0

init = tf.global_variables_initializer()

saver = tf.train.Saver()

train = pd.read_csv('./sign_mnist_train/sign_mnist_train.csv')
train_images = []
train_labels = []
for rows in range(27455):
    image = train.values[rows][1 : ]
    train_images.append(image)
    one_hot_label = [0 for i in range(25)]
    one_hot_label[train.values[rows][0]] = 1
    train_labels.append(one_hot_label)
    
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, './neighbor_model.ckpt')
    while True:
        ret, frame = cap.read()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))
        reshaped = np.reshape(resized, (784,))
        
        output = sess.run(prediction, feed_dict = {x : train_images,
                                                   centroid : reshaped})
        prediction_ = np.argmax(train_labels[output])
        cv2.putText(frame, "{}".format(prediction_), (0, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)

        cv2.imshow('frame', frame)
        
cap.release()
cv2.destroyAllWindows()


    cv2.imshow('frame1', fgmask)
    
    images.append(fgmask)
