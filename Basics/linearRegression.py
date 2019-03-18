import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

## initializing all values

## random values are assigned to lists

xvalues = [random.randint(1, 10) for i in range(5)]

yvalues = [random.randint(1, 10) for i in range(5)]

## setup arrays for plotting

y2values = []

m_c = [[], []]

epoch_amnt = []

## set x & y values, can uncomment if wanted

##xvalues = [1, 2, 3, 4, 5]

##yvalues = [2, 3, 4, 5, 6]

## all need to be tf.float32 because the step size of the optimizer is a decimal

x = tf.placeholder(tf.float32, name = "x")

y = tf.placeholder(tf.float32, name = "y")

## tf.Variable() data types are changed throughout the program,
## tf.get_variable() could also be used

m = tf.Variable(0.5, dtype = tf.float32)

c = tf.Variable(0.5, dtype = tf.float32)

y2 = tf.add(tf.multiply(x, m), c)

## defining loss function

squared_loss = tf.square(y2-y)

loss = tf.reduce_sum(squared_loss)

## optimizer, step is 0.0001, could change for accuracy
optimizer = tf.train.GradientDescentOptimizer(0.0001)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    ## epoch amount is 1000, could change for accuracy 
    ## wouldn't want to overflow the float value
    
    for i in range(1000):
        ## feeding the training process
        
        output = sess.run(train, feed_dict = {x:xvalues, y:yvalues})

        wb = sess.run([m, c])
        m2 = sess.run(m)
        c2 = sess.run(c)

        m_c[0].append(m2)
        m_c[1].append(c2)

        epoch_amnt.append(i)

    ## plotting
    
    y2values.append(sess.run(y2, feed_dict = {x:xvalues, y:yvalues}))

    substitution = np.asarray(xvalues)

    y2values = np.asarray(y2values)

    y2values = y2values.reshape(5,)

    ## plotting slope value

    plt.subplot(321)
    
    plt.plot(epoch_amnt, m_c[0], c = "yellow")

    ## plotting constant value

    plt.plot(epoch_amnt, m_c[1], c = "purple")

    plt.subplot(322)
    
    plt.plot(np.asarray(substitution), np.asarray(y2values), c = "red")

    plt.scatter(xvalues, y2values, c = "green")
    
    plt.scatter(xvalues, yvalues, c = "blue")

    plt.show()

    print(wb, output)
