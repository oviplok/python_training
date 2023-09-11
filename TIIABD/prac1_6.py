a = [1, 2, 3, 4, 2, 1, 3, 4, 5, 6, 5, 4, 3, 2]
b = ['a', 'b', 'c', 'c', 'c', 'b', 'a', 'c', 'a', 'a', 'b', 'c', 'b', 'a']
d = {'a': 0, 'b': 0, 'c': 0}

if __name__ == '__main__':
    for i in range(len(b)):
        if b[i] == 'a':
            d['a'] += a[i]
        elif b[i] == 'b':
            d['b'] += a[i]
        elif b[i] == 'c':
            d['c'] += a[i]

    print("Dictionary:", d)
