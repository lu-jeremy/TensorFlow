import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train = pd.read_csv(r'..\NearestNeighbor\sign_mnist_train\sign_mnist_train.csv')
    
labels = []

images = []
for rows in range(27455):
    train_image = train.values[rows][1 : ]
    images.append(train_image)
    one_hot_label = [0 for i in range(25)]
    one_hot_label[train.values[rows][0]] = 1
    labels.append(one_hot_label)


x = tf.placeholder(tf.float32, shape = [None, 784])
y = tf.placeholder(tf.float32, shape = [None, 25])
w = tf.Variable(np.zeros((784, 25)), dtype = tf.float32)
b = tf.Variable(np.zeros(25), dtype = tf.float32)

prediction = tf.nn.softmax(tf.add(tf.matmul(x, w), b))

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction),
                                     reduction_indices = 1))
# 46.11%
learning_rate = 0.00001
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(20):
        sess.run(train, feed_dict = {x : images, y : labels})
        loss_value = sess.run(loss, feed_dict = {x : images,
                                                 y : labels})
        print(loss_value)

    saver.save(sess, 'train_model.ckpt')

    correct_prediction = tf.equal(tf.argmax(prediction, 1),
                                  tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_value = sess.run(accuracy, feed_dict = {x : images, y : labels})

    print(accuracy_value * 100, '%')
                                  
            


