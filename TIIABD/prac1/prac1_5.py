import string


if __name__ == '__main__':
    numbs = []
    sum_numbs = 0

    while True:
        num = int(input("number: "))
        numbs.append(num)
        sum_numbs += num

        if sum_numbs == 0:
            break

    #Вот это я уже нашел
    sum_squares = sum([n ** 2 for n in numbs])
    print("sum:", sum_squares)


