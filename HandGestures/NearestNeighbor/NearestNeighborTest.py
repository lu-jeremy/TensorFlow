import tensorflow as tf
import numpy as np
import cv2
import time
import pandas as pd

# opencv
cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorKNN()

# tensorflow
tf.reset_default_graph()

trainX = tf.placeholder(tf.float32, shape = [None, 784])
centroid = tf.placeholder(tf.float32, 784)
distance = tf.reduce_sum(tf.abs(tf.add(trainX, tf.negative(centroid))),
                         reduction_indices = 1)
prediction = tf.arg_min(distance, 0)

_ = tf.Variable(initial_value = 'fake_variable')

accuracy = 0

init = tf.global_variables_initializer()

saver = tf.train.Saver()

train = pd.read_csv('./sign_mnist_train/sign_mnist_train.csv')
train_images = []
train_labels = []
for rows in range(27455):
    image = train.values[rows][1 : ]
    train_images.append(image)
    one_hot_label = [0 for i in range(25)]
    one_hot_label[train.values[rows][0]] = 1
    train_labels.append(one_hot_label)
    
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, './neighbor_model.ckpt')
    while True:
        ret, frame = cap.read()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

##        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

##        blur = cv2.GaussianBlur(gray, (5, 5), 0)
##        ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

##        contours, _ = cv2.findContours(thresh1, cv2.RETR_TREE,
##                                       cv2.CHAIN_APPROX_SIMPLE)

        kernel = np.ones((5,5),np.uint8)

        fgmask = fgbg.apply(frame)

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        fgmask = cv2.GaussianBlur(fgmask, (5, 5), 100)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        try:
            cnt = max(contours, key = cv2.contourArea)

            epsilon = 0.0005*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)

            hull = cv2.convexHull(cnt)

            cv2.drawContours(frame, [hull], 0, (0, 255, 0), 2)
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

            x, y, w, h = cv2.boundingRect(approx)

            crop_img = frame[y : y + h, x : x + w]
            resized = cv2.resize(crop_img, (28, 28))
            resized2 = resized[:, :, 0]
            reshaped = np.reshape(resized2, (784,))

            cv2.imshow('crop', fgmask)
            
        except Exception as E:
            print('error', E)
            
        output = sess.run(prediction, feed_dict = {trainX : train_images,
                                                   centroid : reshaped})
        prediction_ = np.argmax(train_labels[output])
        
        cv2.putText(frame, "Prediction: {}".format(prediction_), (0, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        cv2.imshow('frame', frame)

    cap.release()
    cv2.destroyAllWindows()

