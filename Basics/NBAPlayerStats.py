import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

year = []

df = pd.read_csv("C:/Users/bluet/Desktop/TensorFlow/" + \
"CSVDatasets/nba-players-stats/Seasons_Stats.csv")

personal_fouls = []

plot_x = []

plot_y = []

loss_ = []

epoch_amt = []

loss_list = []

for var in range(1950, 2018, 1):
    data_fouls = df['PF'][df['Year'] == var].sum(axis = 0)

    personal_fouls.append(data_fouls)
    
    year.append(var)

x = tf.placeholder(tf.float32, name = 'x')

y = tf.placeholder(tf.float32, name = 'y')

m = tf.Variable(0.1, dtype = tf.float32)

c = tf.Variable(0.1, dtype =  tf.float32)

init_y = tf.add(tf.multiply(m, x), c)



absolute_loss = init_y-y

squared_loss = tf.square(absolute_loss)

loss = tf.reduce_sum(squared_loss)

learning_rate_choices = [0.1, 1, 100, 1000]

for learning_rate in learning_rate_choices:
    optimizer = tf.train.AdamOptimizer(learning_rate)

    train = optimizer.minimize(loss)


    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        
        sess.run(init)

        for epoch in range(4000):

            output = sess.run(train, feed_dict = {x : year, y : personal_fouls})

            m_ = sess.run(m, feed_dict = {x : year, y : personal_fouls})

            c_ = sess.run(c, feed_dict = {x : year, y : personal_fouls})

            epoch_amt.append(epoch)

            loss_.append(sess.run(loss, feed_dict = {x : year, y : personal_fouls}))

        loss_list.append(sess.run(loss, feed_dict = {x : year, y : personal_fouls}))
        
        plt.subplot(321)
        
        plt.scatter(year, personal_fouls)

        for x_ in range(1950, 2050, 1):
            y_ = m_ * x_ + c_
            plot_x.append(x_)
            plot_y.append(y_)

        plt.plot(plot_x, plot_y)

        plt.subplot(322)

        plt.plot(epoch_amt, loss_)
        
        plt.show()

        print([m_, c_])

##        print(sess.run(loss, feed_dict = {x : year, y : personal_fouls}))

plt.plot(learning_rate_choices, loss_list)

plt.show()
    
