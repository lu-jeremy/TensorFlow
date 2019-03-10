import numpy as np

# 1
a = np.array([1, 5, 3])

print(a)

# 2
for i in a.data:
     print(i)

# 3
zero = np.zeros((4, 5))
flattened = zero.ravel()
for index, element in enumerate(flattened.data):
     userGiven = int(input("Give values for the array \n"))
     flattened[index] = userGiven
modified = zero.reshape(4, 5)
output = flattened.sum()
print(modified)

# 4
output = flattened.sum()
print("Output: ", output)

# 5
evenlySpaced = np.linspace(30, 40, 20)
print(evenlySpaced)

# 6
thirtyElements = np.arange(0, 30, 1)
new = thirtyElements.reshape(5, 6)
print(new)

# 7
tenBythree = new.reshape(10, 3)
flattenNew = new.ravel()
print(tenBythree, flattenNew)

# 8
sumNew = new.sum()
print(sumNew)

# 9
print(np.max(new))
print(np.min(new))

