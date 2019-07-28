import tensorflow as tf
import cv2
from imutils import paths
import numpy as np

labels = []

images = []

for image_path in paths.list_images('./gesture_training'):
    
    image = image_path.split('/')[-1]

    image = cv2.imread(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (50, 50))

    image = np.reshape(image, (2500,))

    images.append(image)

    if 'gesture' in image_path:
        labels.append([0, 1])
    elif 'image' in image_path:
        labels.append([1, 0])
        
trainX = tf.placeholder(tf.float32, shape = [None, 2500])

trainY = tf.placeholder(tf.float32, shape = [None, 2])

w = tf.Variable(np.zeros((2500, 2)), dtype = tf.float32)

b = tf.Variable(np.zeros(2), dtype = tf.float32)

prediction = tf.nn.softmax(tf.add(tf.matmul(trainX, w), b))

loss = tf.reduce_mean(-tf.reduce_sum(trainY * tf.log(prediction),
                                     reduction_indices = 1))

learning_rate = 0.0000001

optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(init)

    for epoch in range(100):

        sess.run(train, feed_dict = {trainX : images, trainY : labels})

        loss_value = sess.run(loss, feed_dict = {trainX : images,
                                                 trainY : labels})

        print(loss_value)

    saver.save(sess, 'train_hand_model.ckpt')

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(trainY, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    accuracy_value = sess.run(accuracy, feed_dict = {trainX : images,
                                                     trainY : labels})

    print(accuracy_value * 100, '%')
