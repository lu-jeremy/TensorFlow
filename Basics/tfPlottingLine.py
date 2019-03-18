import tensorflow as tf
import matplotlib.pyplot as plt

## define variables, constants, and placeholders at the start

## define relationships for the above/creating the graph

## run the session and pass the values to the placeholders

## finally, plot or use the results

x = tf.placeholder(tf.uint8)

m = tf.constant(3, tf.uint8)

c = tf.constant(2, tf.uint8)

y = tf.add(tf.multiply(m, x), c)

valuesX = [1, 2, 3, 4, 5, 6]

with tf.Session() as sess:
##    for x2 in range(100):
    var = sess.run(y, feed_dict = {x:valuesX})
    plt.plot(valuesX, var)
    plt.show()
    
        
        
    
        
