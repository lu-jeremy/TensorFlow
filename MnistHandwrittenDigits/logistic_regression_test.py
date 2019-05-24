import tensorflow as tf
import numpy as np
import cv2
from tensorflow.examples.tutorials.mnist import input_data
import time

'''
Idea:
put images and frames into a list
then use those as test images
also try to use hog
not able to do labels for test images 
'''

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

images = []

while True:
    ret, frame = cap.read()

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    print('Say "Cheese"! Took a picture!')
    time.sleep(3)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    reshaped = np.reshape(resized, (784,))
    images.append(reshaped)
    
cap.release()
cv2.destroyAllWindows()

tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape = [None, 784])
y = tf.placeholder(tf.float32, shape = [None, 10])
w = tf.Variable(np.zeros((784, 10)), dtype = tf.float32)
b = tf.Variable(np.zeros(10), dtype = tf.float32)
prediction = tf.nn.softmax(tf.add(tf.matmul(x, w), b))
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction),
                                     reduction_indices = 1))

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    saver.restore(sess, './test_model.ckpt')
    
    output = sess.run(prediction, feed_dict = {x : images})

    
                               
                                        
    

