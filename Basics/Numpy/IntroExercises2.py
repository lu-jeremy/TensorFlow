import numpy as np

# 1
a = np.floor(10 * np.random.random((3, 3)))

b = np.floor(10 * np.random.random((3, 3)))
print(np.vstack((a, b)), "~~", np.hstack((a, b)))

# 2
stacked = np.vstack((a, b))
print(stacked.sum())

# 3
c = np.vstack((a, b))
print(np.hsplit(c, 3))
