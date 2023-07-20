#ну не совсем то что нужно
def addTwoNumbers(l1, l2):
    l3 = []
    l1.reverse()
    l2.reverse()
    ls1 =[str(integer) for integer in l1]
    ls2 = [str(integer) for integer in l2]
    s1 = ''.join(ls1)
    s2 = ''.join(ls2)
    i1 = int(s1)
    i2 = int(s2)
    i3 = i1 + i2
    s3 = str(i3)
    for i in range(len(s3)):
        l3.append(s3[i])
    l3 = [int(string) for string in l3]
    l3.reverse()
    return l3


if __name__ == '__main__':
    print(addTwoNumbers([2, 4, 3], [5, 6, 4]))
