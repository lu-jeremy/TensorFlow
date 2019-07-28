import cv2
import tensorflow as tf
import numpy as np

cap = cv2.VideoCapture(0)

tf.reset_default_graph()

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

with tf.Session()as sess:
    sess.run(init)
    saver.restore(sess, 'train_hand_model.ckpt')
    while True:
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (50, 50))

        image = np.reshape(image, (2500,))

        output = sess.run(prediction, feed_dict = {trainX : [image]})

        print(output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow('frame', frame)
    cap.release()
    cv2.destroyAllWindows()
