import tensorflow as tf
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorKNN()

x = tf.placeholder(tf.float32, shape = [None, 784])
y = tf.placeholder(tf.float32, shape = [None, 25])
w = tf.Variable(np.zeros((784, 25)), dtype = tf.float32)
b = tf.Variable(np.zeros(25), dtype = tf.float32)
prediction = tf.nn.softmax(tf.add(tf.matmul(x, w), b))

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction),
                                     reduction_indices = 1))

saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, 'train_model.ckpt')
    while True:
        ret, frame = cap.read()

        kernel = np.ones((5,5),np.uint8)

        fgmask = fgbg.apply(frame)

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

        fgmask = cv2.GaussianBlur(fgmask, (5, 5), 100)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        try:
            cnt = max(contours, key = cv2.contourArea)

            epsilon = 0.0005 * cv2.arcLength(cnt, True)
            
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            hull = cv2.convexHull(cnt)

            x, y, w, h = cv2.boundingRect(approx)

            cv2.rectangle(frame,(x, y),(x + w,y + h), (0, 255, 0), 2)

            crop_img = frame[y : y + h, x : x + w]
            resized = cv2.resize(crop_img, (28, 28))
            resized2 = resized[:, :, 0]
            reshaped = np.reshape(resized2, (784,))

        except Exception as E:
            print('Error:', E)
        
        cv2.putText(frame, "Prediction: {}".format(output), (0, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            
        cv2.imshow('frame', frame)
        
    cap.release()
    cv2.destroyAllWindows()
