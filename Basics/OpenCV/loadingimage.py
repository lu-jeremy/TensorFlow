import numpy as np
import cv2

img_path = "C:\\Users\\bluet\\Desktop\\Images\\Backgrounds\\shinjuku.png"
img = cv2.imread(img_path)
WINDOW_SIZE = 5

#Initialize Scaled empty image
zeros = np.zeros((len(img)//WINDOW_SIZE,len(img[0])//WINDOW_SIZE),dtype= np.uint8)

##Conversion to Gray
for j in range(0,len(img),1):
    for i in range(0,len(img[0]),1):
        color = img[j][i]
        avg = (color[0]//3+color[1]//3+color[2]//3)
        img[j][i]=avg

##Sliding window of        
for j in range(0,len(img)-WINDOW_SIZE,WINDOW_SIZE):
    for i in range(0,len(img[0])-WINDOW_SIZE,WINDOW_SIZE):
        win_avg = 0
        for y in range(j,j+WINDOW_SIZE,1):
            for x in range(i,i+WINDOW_SIZE,1):
                win_avg+=img[y][x]//(WINDOW_SIZE**2)
        #print(zeros[j//5][i//5],win_avg)
        zeros[j//WINDOW_SIZE][i//WINDOW_SIZE]=win_avg[0]

cv2.imshow("Window", zeros)

while True:
    if (cv2.waitKey(33) & 0xff) == 27:
        break

cv2.destroyAllWindows()

##sumVar = 0
##zeros = np.zeros((1920//5, 1080//5), dtype = np.uint8)
##
##mImage = cv2.imread("C:\\Users\\bluet\\Desktop\\Images\\Backgrounds\\shinjuku.png")
##
##for x in range(len(mImage[0])):
##    for y in range(len(mImage)):
##        color = mImage[y][x]
##        avg = (int(color[0]) + int(color[1]) + int(color[2]))/3
##        mImage[y][x] = avg
##        
##for x2 in range(0, len(mImage[0]), 5):
##    for y2 in range(0, len(mImage), 5):
##        sumVar = 0
##        for x in range(x2, x2 + 5, 1):
##            for y in range(y2, y2 + 5, 1):
##                sumVar += mImage[x][y]
##        
##        avg = sumVar/4
##        zeros[x2//20][y2//20] = avg
##
##cv2.imshow("mWindow", zeros)
 
