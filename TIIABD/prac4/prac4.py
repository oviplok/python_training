import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib

matplotlib.use('TkAgg')  # Use the TkAgg backend
import matplotlib.pyplot as plt


def part1():
    street = np.array([80, 98, 75, 91, 78])
    garage = np.array([100, 82, 105, 89, 102])

    street_list = street.tolist()
    garage_list = garage.tolist()

    corr = np.corrcoef(street, garage)[0, 1]

    print(corr)

    plt.scatter(street_list, garage_list)
    plt.xlabel('Улица')
    plt.ylabel('Гараж')
    plt.title('Диаграмма рассеяния для переменных "Улица" и "Гараж"')
    plt.show()


def part2():
    df = pd.read_csv('insurance.csv')
    df.describe()
    non_numeric = ['sex', 'smoker', 'region']
    df_numeric = df.drop(non_numeric, axis=1)
    corr = df_numeric.corr()
    # Матрица корреляции
    print(corr)

    x = df['age']
    y = df['charges']

    # Среднее x и y
    mean_x = x.mean()
    mean_y = y.mean()
    # Вычисление разницы между x и средним x и y и средним y
    diff_x = x - mean_x
    diff_y = y - mean_y
    # Вычислите наклон (slope) и сдвиг (intercept)
    slope = (diff_x * diff_y).sum() / (diff_x ** 2).sum()
    intercept = mean_y - slope * mean_x
    # Вычислите прогнозные значения y
    predicted_y = slope * x + intercept
    # Вычислите среднеквадратичную ошибку (MSE)
    mse = mean_squared_error(y, predicted_y)
    # Выведите наклон, сдвиг и MSE
    print("Наклон (slope):", slope)
    print("Сдвиг (intercept):", intercept)
    print("Среднеквадратичная ошибка (MSE):", mse)

    # Вычислите наклон и сдвиг регрессионной линии
    slope, intercept = np.polyfit(x, y, 1)
    # Вычислите прогнозные значения y
    predicted_y = slope * x + intercept
    # Создайте график
    plt.scatter(x, y, label='Данные')
    plt.plot(x, predicted_y, color='r', label='Регрессионная линия')
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.title('Регрессия между Age и Charges')
    plt.legend()
    # Отобразите график
    plt.show()


if __name__ == '__main__':
    part2()
