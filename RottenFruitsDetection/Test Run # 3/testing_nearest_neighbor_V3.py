import tensorflow as tf
import numpy as np
import cv2
import time
from imutils import paths

#set up

fgbg = cv2.createBackgroundSubtractorMOG2()

train_images = []
train_labels = []

def path_images(folder_name, image_array, label_array, word, array):
    try:
        for image_path in paths.list_images(folder_name):
            
            image = image_path.split('/')[-1]

            image = cv2.imread(image)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            image = cv2.resize(image, (50, 50))

            image = np.reshape(image, (2500,))

            image_array.append(image)

            if word in image_path:
                label_array.append(array)
    except Exception as E:
        print('Warning:', E)
            
path_images('train', train_images, train_labels, 'apple', [0, 1])
path_images('rotten', train_images, train_labels, 'rotten', [1, 0])

# opencv
cap = cv2.VideoCapture(0)

# tensorflow
tf.reset_default_graph()

trainX = tf.placeholder(tf.float32, shape = [None, 2500])
centroid = tf.placeholder(tf.float32, 2500)
distance = tf.reduce_sum(tf.abs(tf.add(trainX, tf.negative(centroid))),
                         reduction_indices = 1)
prediction = tf.arg_min(distance, 0)

_ = tf.Variable(initial_value = 'fake_variable')

accuracy = 0

init = tf.global_variables_initializer()

saver = tf.train.Saver()
    
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, './neighbor_model.ckpt')
    while True:
        ret, frame = cap.read()

        kernel = np.ones((5, 5), np.uint8)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        fgmask = fgbg.apply(frame)

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
                
        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        try:
            cnt = max(contours, key = cv2.contourArea)

            epsilon = 0.0005*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)

##            hull = cv2.convexHull(cnt)

##            cv2.drawContours(frame, [hull], 0, (0, 255, 0), 2)
            
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

            x, y, w, h = cv2.boundingRect(approx)

            crop_img = frame[y : y + h, x : x + w]

            resized = cv2.resize(crop_img, (50, 50))
            resized2 = resized[:, :, 0]
            reshaped = np.reshape(resized2, (2500,))

            cv2.putText(crop_img, "Your Apple",
            (0, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 1)

            cv2.imshow('cropped', crop_img)

        except Exception as E:
            print('Warning:', E)
            
        output = sess.run(prediction, feed_dict = {trainX : train_images,
                                                   centroid : reshaped})
        print(output)
        prediction_ = np.argmax(train_labels[output])

        if (prediction_ == 0):
            correct_prediction = 'Rotten!'
        elif (prediction_ == 1):
            correct_prediction = 'Not rotten!'
        
        cv2.putText(frame, "Prediction : {}".format(correct_prediction),
                    (0, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 1)

        cv2.imshow('frame', frame)

        
    cap.release()
    cv2.destroyAllWindows()
