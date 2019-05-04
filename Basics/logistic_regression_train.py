import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# import dataset
from tensorflow.examples.tutorials.mnist import input_data

# extracting data from set
mnist_data = input_data.read_data_sets("/tmp/data", one_hot = True)

# set up variables

x = tf.placeholder(tf.float32, shape = [None, 784])

y = tf.placeholder(tf.float32, shape = [None, 10])

w = tf.Variable(np.zeros((784, 10)), dtype = tf.float32)

b = tf.Variable(np.zeros(10), dtype = tf.float32)

prediction = tf.nn.softmax(tf.add(tf.matmul(x, w), b))

# set up saver for model
saver = tf.train.Saver()


# optimizing

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction),
                                     reduction_indices = 1))

learning_rate = 0.002

# learning rate is in the paranthesis
optimizer = tf.train.AdamOptimizer(learning_rate)

# if loss is smaller, then it will be better 
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# main loop
with tf.Session() as sess:
    writer = tf.summary.FileWriter('graphs_logistic', sess.graph)
        
    sess.run(init)

    # specify 10 epochs 
    for epoch in range(10):
        print(epoch)
        average_loss =  0

        # take batches
        batches = int(mnist_data.train.num_examples/100)

        # go through num of batces
        for batch in range(batches):
            # train 100 batches at a time
            batch_x, batch_y = mnist_data.train.next_batch(100)

            sess.run(train, feed_dict = {x : batch_x, y : batch_y})

            loss_value = sess.run(loss, feed_dict = {x : batch_x, y : batch_y})

            average_loss += loss_value
            
        # calculate average loss
        average_loss = average_loss/batches

        print(average_loss)

    w_b_values = sess.run([w, b])

    print(w_b_values)

    # tf.argmax() used to see if indices are equal 
    correct_prediction = tf.equal(tf.argmax(prediction, 1),
                                  tf.argmax(y, 1))

    # tf.reduce_mean() adds up all squared values then square root them
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    accuracy_value = sess.run(accuracy, feed_dict = {x : mnist_data.test.images,
             y : mnist_data.test.labels})

    print(accuracy_value * 100, '%')

    saver.save(sess, 'test_model')

    writer.close()
