import matplotlib.pyplot as plt
import csv

##take NASA data set lol
##use image data set

number_of_diseases_total = 0
diseases_names = []
occurrences = []
percentages = []
explode = []

with open("C:/Users/bluet/Desktop/TensorFlow/CSVDatasets/U.S._Chronic_Disease_Indicators__CDI_.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[5] == "Topic":
            continue
        if row[0] == "2016":
            number_of_diseases_total += 1
            if row[5] not in diseases_names:
                diseases_names.append(row[5])
                occurrences.append(1)
            else:
                index = diseases_names.index(row[5])
                occurrences[index] += 1
    for number in occurrences:
        percentages.append(round((number/number_of_diseases_total) * 100, 2))
    for p in percentages:
        maximum = max(percentages)
        if  maximum == p:
            explode.append(0.1)
        else:    
            explode.append(0)

fig1, ax1 = plt.subplots()

ax1.pie(percentages, labels = diseases_names, explode = explode, autopct = "%1.1f%%", shadow = True, startangle = 90)


ax1.axis("equal")

plt.show()

##print(number_of_diseases_total)
##print(diseases_names)
##print(occurrences)

