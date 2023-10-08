import matplotlib
from scipy import rand

matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn import datasets, __all__, preprocessing
from sklearn.datasets import fetch_openml, load_digits
import csv
from sklearn.manifold import TSNE
from sklearn import manifold, datasets
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import umap
# import umap.plot as umap_plot
import time
import os


def part1To3():
    data = pd.read_csv('insurance.csv')
    print(data)
    print()
    # Вывод статистики по данным
    statistics = data.describe()
    print(statistics)

    # non_numeric = ['smoker', 'region', 'sex']
    # data_numeric = data.drop(non_numeric, axis=1)
    data.hist()
    plt.show()


def part4():
    data = pd.read_csv('insurance.csv')

    # Расчет мер центральной тенденции и мер разброса для bmi
    bmi_mean = data['bmi'].mean()
    bmi_median = data['bmi'].median()
    bmi_std = data['bmi'].std()

    # Расчет мер центральной тенденции и мер разброса для charges
    charges_mean = data['charges'].mean()
    charges_median = data['charges'].median()
    charges_std = data['charges'].std()

    # Вывод результатов
    print(f"BMI: Среднее = {bmi_mean}, Медиана = {bmi_median}, Стандартное отклонение = {bmi_std}")
    print(f"Charges: Среднее = {charges_mean}, Медиана = {charges_median}, Стандартное отклонение = {charges_std}")

    # Построение гистограммы для индекса массы тела (bmi)
    plt.hist(data['bmi'], bins=30, edgecolor='black')
    plt.axvline(bmi_mean, color='red', linestyle='dashed', label='Среднее')
    plt.axvline(bmi_median, color='green', linestyle='dashed', label='Медиана')
    plt.axvline(bmi_mean - bmi_std, color='orange', linestyle='dashed', label='Сред - Откл')
    plt.axvline(bmi_mean + bmi_std, color='orange', linestyle='dashed', label='Сред + Откл')
    plt.legend()
    plt.xlabel('BMI')
    plt.ylabel('Count')
    plt.title('Гистограмма BMI')
    plt.show()

    # Построение гистограммы для расходов (charges)
    plt.hist(data['charges'], bins=30, edgecolor='black')
    plt.axvline(charges_mean, color='red', linestyle='dashed', label='Среднее')
    plt.axvline(charges_median, color='green', linestyle='dashed', label='Медиана')
    plt.axvline(charges_mean - charges_std, color='orange', linestyle='dashed', label='Сред - Откл')
    plt.axvline(charges_mean + charges_std, color='orange', linestyle='dashed', label='Сред + Откл')
    plt.legend()
    plt.xlabel('Charges')
    plt.ylabel('Count')
    plt.title('Гистограмма Charges')
    plt.show()


# todo
def part5():
    plt.figure(figsize=(8, 8))
    data = pd.read_csv('insurance.csv')
    # Построение box-plot
    plt.boxplot([data['age'], data['bmi'], data['charges'], data['children']],
                labels=['age', 'bmi', 'charges', 'children'], vert=False)
    # plt.xticks(np.arange(0, 105, 5))
    # Задание названия графиков
    plt.title('Box-plot for Numerical Features')

    plt.grid()
    # Отображение графика
    plt.show()


def part6():
    data = pd.read_csv('insurance.csv')
    # Set the number of samples and different lengths
    num_samples = 300
    sample_lengths = [10, 50, 100, 500]

    # Choose the feature (charges or bmi)
    feature = 'charges'
    # feature = 'bmi'

    # Initialize an array to store the sample means
    sample_means = []

    # Generate multiple samples and calculate the means
    for length in sample_lengths:
        means = []
        for _ in range(num_samples):
            # Randomly select a sample of length 'length' from the feature
            sample = np.random.choice(data[feature], size=length, replace=False)
            # Calculate the mean of the sample and add it to the list
            mean = np.mean(sample)
            means.append(mean)
        sample_means.append(means)

    # Plot histograms of the sample means
    fig, axs = plt.subplots(len(sample_lengths), 1, figsize=(8, 10))
    for i, means in enumerate(sample_means):
        axs[i].hist(means, bins=30)
        axs[i].set_title(f'Sample Length: {sample_lengths[i]}')

    plt.tight_layout()
    plt.show()

    # Calculate the standard deviation and mean of the sample means
    std_devs = [np.std(means) for means in sample_means]
    means = [np.mean(means) for means in sample_means]

    # Print the results
    print(f'Standard Deviations: {std_devs}')
    print(f'Means: {means}')


def part7():
    data = pd.read_csv('insurance.csv')

    charges_mean = data['charges'].mean()
    charges_std = data['charges'].std()

    bmi_mean = data['bmi'].mean()
    bmi_std = data['bmi'].std()

    charges_n = data['charges'].shape[0]
    bmi_n = data['bmi'].shape[0]

    charges_critical_value_95 = stats.t.ppf(0.975, df=charges_n - 1)
    charges_critical_value_99 = stats.t.ppf(0.995, df=charges_n - 1)

    bmi_critical_value_95 = stats.t.ppf(0.975, df=bmi_n - 1)
    bmi_critical_value_99 = stats.t.ppf(0.995, df=bmi_n - 1)

    charges_margin_error_95 = charges_critical_value_95 * (charges_std / (charges_n ** 0.5))
    charges_margin_error_99 = charges_critical_value_99 * (charges_std / (charges_n ** 0.5))

    bmi_margin_error_95 = bmi_critical_value_95 * (bmi_std / (bmi_n ** 0.5))
    bmi_margin_error_99 = bmi_critical_value_99 * (bmi_std / (bmi_n ** 0.5))

    charges_confidence_interval_95 = (charges_mean - charges_margin_error_95, charges_mean + charges_margin_error_95)
    charges_confidence_interval_99 = (charges_mean - charges_margin_error_99, charges_mean + charges_margin_error_99)

    bmi_confidence_interval_95 = (bmi_mean - bmi_margin_error_95, bmi_mean + bmi_margin_error_95)
    bmi_confidence_interval_99 = (bmi_mean - bmi_margin_error_99, bmi_mean + bmi_margin_error_99)

    print("Confidence Intervals for Charges:")
    print("95% Confidence Interval:", charges_confidence_interval_95)
    print("99% Confidence Interval:", charges_confidence_interval_99)

    print("\nConfidence Intervals for BMI:")
    print("95% Confidence Interval:", bmi_confidence_interval_95)
    print("99% Confidence Interval:", bmi_confidence_interval_99)


if __name__ == '__main__':
    # print("choose part(1 or 4 or 5 or 6):")
    # x = int(input())
    # if x == 1:
    #     part1To3()
    # if x == 4:
    #     part4()
    # if x == 5:
    #     part5()
    # if x == 6:
    #     part6()
    # if x == 7:
    part7()
