import numpy as np
import cv2


"""
[0, 0, (1 & 2)
0, 0, (3 & 4)
0, 0, (5 & 6)
0] (7)

4 & 5 need to be 0.

Needs to go through rows and columns in order to actually position the black
sqr correctly

in order to go through rows and columns, you can use np.where() to find the
specific indices of where the elements I want to replace with 0 are


since this is a 400 x 400 plain, I have to make sure the indices are in bounds
of 180 and 220, because since the square is 20 x 20 it's apparent that I would
have to make it exactly in the middle such that when I draw a line through the
middle, the line is a line of symmetry that divides the square into two equal
parts; so 200 - 20 = 180 & 200 + 20 = 220

however one problem I encountered down the way was that when I had a tuple
with two arrays, one with row indices and the other one with column indices,
and after I attempted to change the elements of the reshaped array using the
indices in the indices array I realized that the original reshaped array
doesn't go by rows and columns; so then if I replaced the elements using the
indices in the indices array I would be only replacing the top few pixels
so i wouldn't know whether the actual program went to the next line in the
reshaped array

400 x 179
i though i could use 400 x 179 or 400 x 221 in order to find the indices
of the elements I would want to replace but apparently the program doesn't
go through the array like that

but instead of doing all that work I just calculated what position the
square would need to be in and replacing those elements in the flattened
array by using those calculations
"""

arr = np.zeros((400, 400), np.uint8)
flattened = arr.ravel()

for index, element in enumerate(flattened.data):
    flattened[index] = 255;

reshaped = flattened.reshape(400, 400)

for x in range(179, 221, 1):
    for y in range(179, 221, 1):
        reshaped[x][y] = 0

##for columns in np.nditer(reshaped):
##    for rows in np.nditer(reshaped):
##        if reshaped.index(columns) > 179 and reshaped.index(columns) < 221:
##            pass
##        if reshaped.index(rows) > 179 and reshaped.index(rows) < 221:
##            pass

##indices_of_rows_columns = np.where(reshaped == 255)
##
##for index in indices_of_rows_columns[1]:
##    if index > 179 and index < 221:
##        print(reshaped[index * 400])
            
cv2.imshow("Window", reshaped)

while True:
    if (cv2.waitKey(33) & 0xff) == 27:
        break

cv2.destroyAllWindows()

