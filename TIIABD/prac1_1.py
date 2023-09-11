import math
import string

d = dict()


def round_fig(x):
    result = math.pi * (x ** 2)
    return {'round', result}


def tri_fig(x, y):
    result = (1 / 2 * x) * y
    return {'triangle', result}


def block_fig(x, y):
    result = x * y
    return {'block', result}


if __name__ == '__main__':
    d = dict
    print("figure: ")
    figure = input()
    answer = 0.0

    # Вызов метода круга
    if figure == "round":
        print("param: ")
        x = int(input())
        answer = (round_fig(x))
        print(answer)

    # Вызов метода квадрата
    if figure == "block":
        print("param 1: ")
        x = int(input())
        print("\nparam 2: ")
        y = int(input())
        answer = (block_fig(x, y))
        print(answer)

    # Вызов метода треугольника
    if figure == "triangle":
        print("param 1: ")
        x = int(input())
        print("\nparam 2: ")
        y = int(input())
        answer = (tri_fig(x, y))
        print(answer)
