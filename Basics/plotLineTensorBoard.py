import tensorflow as tf
import matplotlib.pyplot as plt

tf.reset_default_graph()

## 

## define variables, constants, and placeholders at the start

## define relationships for the above/creating the graph

## run the session and pass the values to the placeholders

## finally, plot or use the results

##x = tf.placeholder(tf.uint8, name = "x")
##
##
##m = tf.constant(3, tf.uint8, name = "m")
##
##c = tf.constant(2, tf.uint8, name = "c")
##
##y = tf.add(tf.multiply(m, x), c)

##x = tf.placeholder(tf.int16, name = "x")
##
##a = tf.constant(9, tf.int16, name = "a")
##
##b = tf.constant(4, tf.int16, name = "b")
##
##c = tf.constant(10, tf.int16, name = "c")

## able to name the operations as well

##y = tf.add(tf.add(tf.multiply(tf.multiply(x, x), a, ), tf.multiply(b, x)), c)
##
##valuesX = [0, 1, 2, 3, 4, 5, 6]

##W = tf.Variable(tf.zeros([784,10]))

tf.get_variable("weight_matrix", shape=(784, 10),
                initializer=tf.zeros_initializer())

a = tf.get_variable(name="var_1", initializer=tf.constant(2))

b = tf.get_variable(name = "var_2", initializer = tf.constant(4))

y = tf.add(a, b)

variable = tf.global_variables_initializer()

with tf.Session() as sess:

    writer = tf.summary.FileWriter("./graphs", sess.graph)

    sess.run(variable)
    a2 = sess.run(a)
    b2 = sess.run(b)
    y2 = sess.run(y)

    print(a2, b2, y2)
        
        
    
        
