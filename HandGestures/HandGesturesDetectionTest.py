import tensorflow as tf
import numpy as np
import pandas as pd

test = pd.read_csv('./sign_mnist_test/sign_mnist_test.csv')

labels = []
images = []
for rows in range(7172):
    test_image = test.values[rows][1 : ]
    images.append(test_image)
    one_hot_label = [0 for i in range(25)]
    labels.append(one_hot_label)
    

x = tf.placeholder(tf.float32, shape = [None, 784])
y = tf.placeholder(tf.float32, shape = [None, 25])
w = tf.Variable(np.zeros((784, 25)), dtype = tf.float32)
b = tf.Variable(np.zeros(25), dtype = tf.float32)
prediction = tf.nn.softmax(tf.add(tf.matmul(x, w), b))
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction),
                                     reduction_indices = 1))
    
init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    saver.restore(sess, './train_model.ckpt')

    correct_prediction = tf.equal(tf.argmax(prediction, 1),
                                  tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_value = sess.run(accuracy, feed_dict = {x : images, y : labels})


##    var = sess.run(prediction, feed_dict = {x : images})

##    print(var)
    
    print(accuracy_value * 100, '%')

    
