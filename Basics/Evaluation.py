##x = 1
##y = 1.0
##z = '1'
##
##print(x, y, z)
##
##user_input = input('hi \n')
##
##if int(user_input) == x:
##    print('integer')
##    if 1 == 1:
##        print('afjweoiowej')
##elif user_input == y:
##    print('float')
##else:
##    print('string')
##    
##for i in range(11):
##    print(i)
##
##string = 'eafnofgwenbi'
##
##for index, i in enumerate(range(len(string))):
##    print(string[i], index)
##
##count = 11
##while (0 <count < 12):
##    count -= 1
##    print(count)
##
##count = 0
##increment = 1
##
##while True:
##    print(count)
##    count += increment
##    if (count == 10):
##        if (increment > 0):
##            increment = -(increment)
##    elif (count == 0):
##        if (increment < 0):
##            increment = -(increment)

##empty_list = ['apple', 'orange', 'YEET', 1]
##
##print(empty_list[1])
##
##empty_list[1] = 'grape'
##
##empty_list.append('LOL')
##
##empty_list.insert(0, ',')
##
##empty_list[4] += 1
##
##empty_list.pop(0)
##
##del empty_list[0]
##
##empty_list.remove(2)
##
##for index, i in enumerate(empty_list):
##    print(i, index)
##
##print(empty_list)

dictionary = {'IDLE' : 23, 'WALK' : 2, 'SPRINT' : 100, 'FLY' : 50}

##for i in dictionary.keys():
##    print(i, dictionary[i])
##    
##dictionary['RUN'] = 3
##
##dictionary['WALK'] = 4
##
##dictionary['RUN'] += 1
##
##dictionary.pop('RUN')
##
##del dictionary['WALK']
##
##print(dictionary)

empty_list = []

for i in dictionary:
    empty_list.append(dictionary[i])

empty_list.sort()

empty_list.reverse()

count = 0
for i in dictionary:
    print(i, empty_list[count])
    count += 1





