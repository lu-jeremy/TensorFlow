import tensorflow as tf
import numpy as np
import cv2
import time

#set up

train_images = []
train_labels = []

def path_images(folder_name, image_array, label_array):
    for image_path in paths.list_images(folder_name):
        
        image = image_path.split('/')[-1]

        image = cv2.imread(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (28, 28))

        image = np.reshape(image, (784,))

        image_array.append(image)

        if 'apple' in image_path:
            label_array.append([0, 1])
        elif 'rotten' in image_path:
            label_array.append([1, 0])
            
path_images('train', train_images, train_labels)

# opencv
cap = cv2.VideoCapture(0)

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
    
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, './neighbor_model.ckpt')
    while True:
        ret, frame = cap.read()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh1, cv2.RETR_TREE,
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

            cv2.imshow('crop', reshaped)
            
        except Exception as E:
            print('error', E)
            
        output = sess.run(prediction, feed_dict = {trainX : train_images,
                                                   centroid : reshaped})
        prediction_ = np.argmax(train_labels[output])
        
        cv2.putText(frame, "{}".format(prediction_), (0, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)

        cv2.imshow('frame', frame)

        
    cap.release()
    cv2.destroyAllWindows()
