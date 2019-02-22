import matplotlib.pyplot as plt
import csv

##plt.pie()
##name_of_diseases and the percentage out of 100
##take NASA data set lol
##use image data set

number_of_diseases_total = 0
diseases_names = []
occurrences = []
percentages = []

with open("C:/Users/bluet/Downloads/U.S._Chronic_Disease_Indicators__CDI_.csv") as f:
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
        
    
##print(number_of_diseases_total)
##print(diseases_names)
##print(occurrences)

