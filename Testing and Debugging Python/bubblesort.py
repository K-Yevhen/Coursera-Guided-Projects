def bubbleSort(listreverse=0):
    lengh = len(list)
    for i in range(0, lengh - 1):
        for j in range(0, lengh - i - 1):
            if list[j] > list[j + 1]:
                list[j], list[j + 1] = list[j + 1], list[j]

    if reverse:
        list.reverse()
    return list
