import numpy as np
import cv2

IMG_SIZE = 5
mSum = 0

mImage = cv2.imread("C:\\Users\\bluet\\Desktop\\Images\\Backgrounds\\shinjuku.png")

zeros = np.zeros((len(mImage[0])//5, len(mImage)//5), np.uint8)

for x in range(0, len(mImage[0]), 1):
    for y in range(0, len(mImage), 1):
        color = mImage[y][x]
        avg = (int(color[0]) + int(color[1]) + int(color[2]))//3
        mImage[y][x] = avg

for rowSquare in range(0, len(mImage[0]) - IMG_SIZE, IMG_SIZE):
    for columnSquare in range(0, len(mImage) - IMG_SIZE, IMG_SIZE):
        for xPixels in range(rowSquare, rowSquare + 5, 1):
            for yPixels in range(columnSquare, columnSquare + 5, 1):
                mSum += mImage[yPixels][xPixels]//(IMG_SIZE**2)
                zeros[rowSquare//IMG_SIZE][columnSquare//IMG_SIZE] = mSum[0]

cv2.imshow("Window", zeros)
