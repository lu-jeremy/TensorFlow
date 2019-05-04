import tensorflow as tf
import numpy as np
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import cv2
from imutils import paths
import matplotlib.pyplot as plt

images = []
_labels = []

for image_path in paths.list_images('test'):
    
    image = image_path.split('/')[-1]

    image = cv2.imread(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (50, 50))

    image = np.reshape(image, (2500,))

    images.append(image)

##print_tensors_in_checkpoint_file('./train_orange_model.ckpt', tensor_name = '',
##                                 all_tensors = True)

tf.reset_default_graph()

trainX = tf.placeholder(tf.float32, shape = [None, 2500])

trainY = tf.placeholder(tf.float32, shape = [None, 2])

w = tf.Variable(np.zeros((2500, 2)), dtype = tf.float32)

b = tf.Variable(np.zeros(2), dtype = tf.float32)


prediction = tf.nn.softmax(tf.add(tf.matmul(trainX, w), b))

loss = tf.reduce_mean(-tf.reduce_sum(trainY * tf.log(prediction),
                                     reduction_indices = 1))


init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(init)

    saver.restore(sess, './train_orange_model.ckpt')

    output = sess.run(prediction, feed_dict = {trainX : images})

    for row in range(10):
        for column in range(1):
            if output[row][column] > output[row][column + 1]:
                _labels.append('orange')
            else:
                _labels.append('apple')
        plt.subplot(6, 2, row + 1)
        image = np.reshape(images[row], (50, 50))
        plt.imshow(image)
        plt.text(0.5, 0.5, '{}'.format(_labels[row]),
                 horizontalalignment = 'right', verticalalignment = 'top')
    plt.show()
                   

    

    




