import cv2

cap = cv2.VideoCapture(0)
 
while True:
    ret, frame = cap.read()
    
    cv2.imshow('frame',gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    

cap.release()
cv2.destroyAllWindows()

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
    sess.init()
    sess.restore(sess, 'train_model.ckpt')

    output = sess.run(prediction, feed_dict = {trainX : test_images})
    print(output.index(1))
