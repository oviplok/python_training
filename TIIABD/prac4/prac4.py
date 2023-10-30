import numpy as np
import statsmodels.stats.multicomp as mc
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.stats import f_oneway
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import ttest_ind
import itertools
from sklearn.preprocessing import MinMaxScaler
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

    mean_x = x.mean()
    mean_y = y.mean()

    diff_x = x - mean_x
    diff_y = y - mean_y
    # наклон (slope) и сдвиг (intercept)
    slope = (diff_x * diff_y).sum() / (diff_x ** 2).sum()
    intercept = mean_y - slope * mean_x
    # прогнозные значения y
    predicted_y = slope * x + intercept
    # среднеквадратичная ошибка (MSE)
    mse = mean_squared_error(y, predicted_y)

    print("Наклон (slope):", slope)
    print("Сдвиг (intercept):", intercept)
    print("Среднеквадратичная ошибка (MSE):", mse)

    # наклон и сдвиг регрессионной линии
    # slope, intercept = np.polyfit(x, y, 1)
    # прогнозные значения y
    predicted_y = slope * x + intercept
    # график
    plt.scatter(x, y, label='Данные')
    plt.plot(x, predicted_y, color='r', label='Регрессионная линия')
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.title('Регрессия между Age и Charges')
    plt.legend()
    plt.show()


def part3():
    data = pd.read_csv('insurance.csv')
    # print(data.head())
    data = data.dropna()  # удалить строки с пропущенными значениями
    # Вывод преобразованных данных
    print(data.head())

    print("\n---Part 3.1--------------------------")
    # Создаем список данных BMI для каждого региона
    region_1_bmi = data[data['region'] == 'southwest']['bmi']
    region_2_bmi = data[data['region'] == 'southeast']['bmi']
    region_3_bmi = data[data['region'] == 'northwest']['bmi']
    region_4_bmi = data[data['region'] == 'northeast']['bmi']

    # ANOVA-тест
    statistic_3_1, p_value_3_1 = f_oneway(region_1_bmi, region_2_bmi, region_3_bmi, region_4_bmi)

    print("Статистика ANOVA:", statistic_3_1)
    print("p-value:", p_value_3_1)

    print("\n---Part 3.2--------------------------")
    model = ols('bmi ~ region', data=data).fit()
    anova_table = anova_lm(model)
    print(anova_table)

    print("\n---Part 3.3--------------------------")
    regions = data['region'].unique()
    region_pairs = list(itertools.combinations(regions, 2))

    alpha = 0.05 / len(region_pairs)  # Определение уровня значимости после поправки Бонферрони

    for pair in region_pairs:
        region_1 = pair[0]
        region_2 = pair[1]

        bmi_region_1 = data[data['region'] == region_1]['bmi']
        bmi_region_2 = data[data['region'] == region_2]['bmi']

        stat, p_value_3_1 = ttest_ind(bmi_region_1, bmi_region_2)

        if p_value_3_1 < alpha:
            print(
                f"Статистически значимая разница в индексе массы тела между регионом {region_1} и регионом {region_2}")

    print("\n---Part 3.4--------------------------")
    F = anova_table['F'][0]
    p_value_3_5 = anova_table['PR(>F)'][0]
    if p_value_3_5 < alpha:
        posthoc = mc.pairwise_tukeyhsd(data['bmi'], data['region'])
        print(posthoc)
        # Plot the results
        sns.pairplot(hue='region', data=data)
        posthoc.plot_simultaneous()
        plt.show()
    else:
        print("Нет статистически значимой разницы между группами.")

    print("\n---Part 3.5--------------------------")
    model_3_5 = ols('bmi ~ region + sex', data=data).fit()
    anova_table_3_5 = anova_lm(model_3_5)

    # Выводим результаты
    print(anova_table_3_5)
    print("\n---Part 3.6--------------------------")
    F_region = anova_table_3_5['F'][0]
    p_value_region = anova_table_3_5['PR(>F)'][0]
    F_sex = anova_table_3_5['F'][1]
    p_value_sex = anova_table_3_5['PR(>F)'][1]

    # статистически значимая разница
    if p_value_region < 0.05 or p_value_sex < 0.05:
        # пост-хок тесты Тьюки
        posthoc_region = mc.MultiComparison(data['bmi'], data['region'])
        posthoc_sex = mc.MultiComparison(data['bmi'], data['sex'])
        result_region = posthoc_region.tukeyhsd()
        result_sex = posthoc_sex.tukeyhsd()

        # Выводим результаты пост-хок тестов
        print("Результаты пост-хок теста для региона:")
        print(result_region)
        print("\nРезультаты пост-хок теста для пола:")
        print(result_sex)

        sns.boxplot(x='region', y='bmi', hue='sex', data=data)
        plt.show()
    else:
        print("Нет статистически значимой разницы между группами.")


if __name__ == '__main__':
    part3()
