'''
- take 10 images from the internet

- find features of all of them, put in loop, use hog_rescaled_image for the output

- use KNeighborsClassifier from sklearn

- prediction (orange or apple)
    - give it final image(s) (11th image or test images) to see whether or
      not it is an orange or not

'''

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure

from sklearn.neighbors import KNeighborsClassifier

import numpy as np

import cv2

from imutils import paths

# initialize matrices

labels = []

hogs = []

images = []

# put all training images into a matrix
def image_path_train():
    
    for image_path in paths.list_images('train'):
        
        image = image_path.split('/')[-1]
        
        images.append(image)
        
# find features of fruits with HOG
def hog_features():

    # go through image matrix
    for image_var in images:

        # read image
        image = cv2.imread(image_var)

        # resize image
        image = cv2.resize(image, (50, 50))

        # grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # specify cell size and find features
        H = hog(image, orientations = 8, pixels_per_cell = (5, 5),
                cells_per_block = (1, 1), transform_sqrt = True,
                block_norm = 'L1')
        
        # put hog images into matrix
        hogs.append(H)

        # say whether a fruit is an apple or not to train
        if 'apple' in image_var:
            labels.append('apple')
        else:
            labels.append('orange')

# # use KNeighborsClassifier for the model, differentiate images
def test_images():

    model = KNeighborsClassifier(n_neighbors = 2)

    model.fit(hogs, labels)

    for (i, image_path) in enumerate(paths.list_images('test')):

        # read image
        image_color = cv2.imread(image_path)

        # resize image
        image = cv2.resize(image_color, (50, 50))

        # grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # find features of test images
        (H, hog_image) = hog(image, orientations = 8, pixels_per_cell = (5, 5),
                            cells_per_block = (1, 1), transform_sqrt = True,
                            block_norm = 'L1', visualize = True)

        # make prediction
        prediction = model.predict(H.reshape(1, -1))[0]

        cv2.namedWindow('Test image #{}'.format(i + 1), cv2.WINDOW_NORMAL)

        cv2.resizeWindow('Test image #{}'.format(i + 1), 600, 600)
                         
        # put prediction results on screen
        cv2.putText(image_color, prediction.title(), (0, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)

        # put image on screen
        cv2.imshow('Test image #{}'.format(i + 1), image_color)

        cv2.waitKey(0)
                    
# call all functions in sequence
def main():
    
    image_path_train()
    
    hog_features()
    
    test_images()


if __name__ == '__main__':
    # call main
    main()



