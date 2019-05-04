##import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

## read

df = pd.read_csv("C:/Users/bluet/Desktop/TensorFlow/CSVDatasets/digits-embedding.csv", names = ["index", "digit", "f1", "f2"])

## setup

def take_average(cluster):
    sum_ = 0
    avg = 0
    for index, num in enumerate(cluster):
        sum_ += num
    avg = sum_/(index+1)

    return avg

c1_x = 0

c1_y = 0

color = ["green", "blue", "red", "yellow", "black", "purple", "magenta", "gray", "cyan", "purple"]

clusterOne_x = []

clusterTwo_x = []

clusterOne_y = []

clusterTwo_y = []

for digit in range(0, 10, 1):
    
    f1 = df["f1"][df["digit"] == digit]

    f2 = df["f2"][df["digit"] == digit]

    plt.scatter(f1, f2, c = color[digit])

plt.show()

random.seed(100)

df10 = df[(df.digit == 0) | (df.digit == 1)]

sample = df10.sample(n = 2)

centroids = sample[["f1", "f2"]].values

plt.plot(centroids[0][0], centroids[0][1], marker = "^", c = "red")
plt.plot(centroids[1][0], centroids[1][1], marker = "^", c = "green")

x = centroids[0][0]

y = centroids[0][1]

x1 = centroids[1][0]

y1 = centroids[1][1]

distance_ = []

data = df[(df["digit"] == 0) | (df["digit"] == 1)].values

for iterations in range(2):
    for var in data:
        
        distance = math.sqrt(((var[2] - x)**2) + ((var[3] - y)**2))
        distance1 = math.sqrt(((var[2] - x1)**2) + ((var[3] - y1)**2))

        if distance > distance1:
            clusterOne_x.append(var[2])
            clusterOne_y.append(var[3])
        else:
            clusterTwo_x.append(var[2])
            clusterTwo_y.append(var[3])

    ## average x and y

    x = take_average(clusterOne_x)

    y = take_average(clusterOne_y)

    x1 = take_average(clusterTwo_x)

    y1 = take_average(clusterTwo_y)

    plt.scatter(x, y, s = 400, marker = "<", c = "magenta")

    plt.scatter(x1, y1, s = 400, marker = "<", c = "blue")

    plt.scatter(clusterOne_x, clusterOne_y, c = "black")

    plt.scatter(clusterTwo_x, clusterTwo_y, c = "yellow")

    plt.show()
