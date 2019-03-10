import matplotlib.pyplot as plt
import csv

year = []
number_of_occurrences_for_each_year = []

with open("C:/Users/bluet/Desktop/TensorFlow/CSVDatasets/U.S._Chronic_Disease_Indicators__CDI_.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[5] == "Topic":
            continue
        if row[5] == "Diabetes":
            if row[0] not in year:
                year.append(row[0])
                number_of_occurrences_for_each_year.append(1)
            else:
                index = year.index(row[0])
                number_of_occurrences_for_each_year[index] += 1
    year.reverse()
    number_of_occurrences_for_each_year.reverse()

plt.plot(year, number_of_occurrences_for_each_year)

plt.show()
