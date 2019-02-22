import matplotlib.pyplot as plt
import csv

#label the graph with the name of diseases
#make comparison with 2016 & 2015
#identify the largest occurrences of a particular disease
#plot increase of all diseases (line graph)
#plot a piechart indicating the total amount of diseases that are occurring the most

topics = []
name_of_diseases_15 = []
number_of_occurrences_15 = []
name_of_diseases_16 = []
number_of_occurrences_16 = []

with open("C:/Users/bluet/Downloads/U.S._Chronic_Disease_Indicators__CDI_.csv") as f:
    reader = csv.reader(f)
    ##going through the rows with the reader
    for row in reader:
        ##if the 5th row has "Topic", then continue because we don't need that
        if row[5] == "Topic":
            continue
        ##checking if diseases are in "2015"
        if row[0] == "2015":
            ##if the disease is not already in the list, then append the disease and it's occurrence as 1
            if row[5] not in name_of_diseases_15:
                name_of_diseases_15.append(row[5])
                number_of_occurrences_15.append(1)
            else:
                ##increment the occurrence for the disease, and find the index of the disease first 
                index_disease = name_of_diseases_15.index(row[5])
                number_of_occurrences_15[index_disease] += 1
        ##repeat the same thing for 2016 over here
        if row[0] == "2016":
            if row[5] not in name_of_diseases_16:
                name_of_diseases_16.append(row[5])
                number_of_occurrences_16.append(1)
            else:
                index_disease = name_of_diseases_16.index(row[5])
                number_of_occurrences_16[index_disease] += 1


plt.bar([(i) for i in range(17)], number_of_occurrences_15, 0.2, label = "2015", color = "r")

plt.bar([(i + 0.2) for i in range(17)], number_of_occurrences_16, 0.2, label = "2016", color = "g")
        
plt.xticks([(i + 0.1) for i in range(17)], name_of_diseases_15, fontsize = 7, rotation = 20)

##plt.bar(i for i in range(17)], year_start, label = "Year Start", color = "b")

plt.legend()

plt.xlabel("Disease/Condition")

plt.ylabel("Number of occurrences for diseases in 2015 & 2016")

plt.show();
