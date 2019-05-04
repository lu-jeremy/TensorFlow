import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

##  setup


## chronic diseases file path :
## C:/Users/bluet/Desktop/TensorFlow/CSVDatasets/U.S._Chronic_Disease_Indicators__CDI_.csv

year = 2011
year_lst = []
num_occurrences = []
xvalues = []
yvalues = []


df = pd.read_csv("C:/Users/bluet/Desktop/TensorFlow/" + \
                         "CSVDatasets/U.S._Chronic_Disease_Indicators__CDI_.csv")


df_tobacco = df[df['Topic'] == 'Tobacco']

for var in range(year, 2017, 1):
    df_year = df_tobacco[df_tobacco['YearStart'] == var]
    year_lst.append(var)
    num_occurrences.append(len(df_year.index))

##plt.scatter(year_lst, num_occurrences)
##
##plt.xlabel('year')
##
##plt.ylabel('number of occurrences')
##
##plt.show()

x = tf.placeholder(tf.float32, name = 'x')

y = tf.placeholder(tf.float32, name = 'y')

m = tf.Variable(0.5, dtype = tf.float32)

c = tf.Variable(0.5, dtype = tf.float32)

initial_y = tf.add(tf.multiply(m, x), c)

absolute_loss = initial_y-y

squared_loss = tf.square(absolute_loss)

loss = tf.reduce_sum(squared_loss)

optimizer = tf.train.AdamOptimizer(0.01)

train = optimizer.minimize(loss)


init = tf.global_variables_initializer()


with tf.Session() as sess:
    
    sess.run(init)

    for epoch in range(2000):
        
        output = sess.run(train, feed_dict = {x:year_lst, y:num_occurrences})

        m1 = sess.run(m, feed_dict = {x:year_lst, y:num_occurrences})

        c1 = sess.run(c, feed_dict = {x:year_lst, y:num_occurrences})

    plt.scatter(year_lst, num_occurrences)

    for i in range(2000, 2050, 5):
        y = m1 * i + c1
        xvalues.append(i)
        yvalues.append(y)

    

    plt.plot(xvalues, yvalues)

    plt.show()

    








