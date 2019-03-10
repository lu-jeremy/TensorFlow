import numpy as np

a = np.floor(10 * np.random.random((2, 2)))

b = np.floor(10 * np.random.random((2, 2)))

vStack = np.vstack((a, b))

hStack = np.hstack((a, b))

print(vStack, "~~", hStack)

randArr = np.floor(10 * np.random.random((2, 12)))

hSplit = np.hsplit(randArr, 3)

hSplit2 = np.hsplit(randArr, (3, 4))

print(hSplit, "~~", hSplit2)
