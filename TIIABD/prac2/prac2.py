import matplotlib
import numpy as np
import pandas as pd
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

matplotlib.use('TkAgg')

count = 0

mnist = fetch_openml("mnist_784", cache=True, parser="liac-arff")
mnist_x = mnist.data
mnist_y = mnist.target.astype(int)


def print_razd(part):
    global count
    count = count + part
    print("\npart " + str(count) + " //////////////////////////////////////////////////////////////////////")


# Загрузка данных из файла CSV
data2 = pd.read_csv('fashion-mnist_train.csv')
data = pd.read_csv('pokemon.csv')


# mnis_x = data2
# mnis_y = data2.target.astype(int)


def part1():
    print_razd(1)
    # Описание найденных данных
    description = data.describe()
    # Вывод описания данных
    print(description)


def part2():
    print_razd(2)
    # Описание c помощью info и head
    info = data.info
    head = data.head()
    # Вывод описания данных
    print(info)
    print()
    print(head)


def part3():
    print_razd(3)
    global data
    x_data = data['type1']
    y_data = data['generation']

    # Создание столбчатой диаграммы
    data = [go.Bar(
        x=x_data,
        y=y_data,
        marker=dict(line=dict(color='black', width=2)),
        width=0.8
    )]

    # Создание макета диаграммы
    layout = go.Layout(
        title=dict(text='Pokemon generation statistic',
                   x=0.5,
                   y=1,
                   xanchor='center',
                   yanchor='top',

                   font=dict(size=20)),

        # Надеюсь правильно, а то не понял
        xaxis=dict(
            title='type',
            tickangle=315,
            gridwidth=2,
            gridcolor='ivory',
            title_font=dict(size=16),
            tickfont=dict(size=14)),

        yaxis=dict(
            title='generation',
            gridwidth=2,
            gridcolor='ivory',
            title_font=dict(size=16),
            tickfont=dict(size=14)),

        bargap=0,
        height=700

    )

    # Создание объекта Figure и отображение диаграммы
    fig = go.Figure(data=data, layout=layout)
    fig.show()


# def part3_2():
#     print_razd()
#     global data
#     x_data = data['type1']
#     y_data = data['generation']
#
#     # Создание столбчатой диаграммы
#     data = [go.Bar(
#         x=x_data,
#         y=y_data,
#     )]
#
#     # Создание макета диаграммы
#     layout = go.Layout(
#         title='Pokemon generation statistic',
#         xaxis=dict(title='type'),
#         yaxis=dict(title='generation')
#     )
#
#     # Создание объекта Figure и отображение диаграммы
#     fig = go.Figure(data=data, layout=layout)
#     fig.show()
#     pass


def part4():
    print_razd(4)
    global data
    x_data = data['type1']
    y_data = data['generation']

    pie_data = [go.Pie(
        labels=x_data,
        values=y_data,
        textinfo='label+percent',
        marker=dict(line=dict(color='white', width=2))  # Specify black lines with width 2
    )]

    # Creating the layout for the pie chart
    layout_pie = go.Layout(
        title='Pie Chart',
        showlegend=False,
        annotations=[dict(text='Categories', x=0.5, y=0.5, font=dict(size=20, color='black'))],
        # pie=dict(hole=0.4, direction='clockwise'),
    )

    # Creating the Figure object and displaying the pie chart
    fig_pie = go.Figure(data=pie_data, layout=layout_pie)
    fig_pie.show()


def part5():
    print_razd(5)
    # Выбор показателя для оси X
    data = pd.read_csv('pokemon.csv')
    x_data = data['name']

    # Выбор показателей для оси Y
    y_data = data['total']
    # y_data_1 = data['hp']
    # y_data_2 = data['attack']
    # y_data_3 = data['defense']
    # y_data_4 = data['sp_attack']
    # y_data_5 = data['sp_defense']
    # y_data_6 = data['speed']

    # Создание маркеров
    plt.scatter(x_data, y_data, color='white', edgecolors='black', linewidth=2)

    # Добавление сетки
    plt.grid(linewidth=2, color='mistyrose')

    # Создание линейных графиков
    plt.plot(x_data, y_data,
             color="crimson",
             markerfacecolor="white",
             markeredgecolor="black",
             markeredgewidth=2,
             linestyle="-",
             label='Total Power')

    # plt.plot(x_data, y_data_1, label='HP')
    # plt.plot(x_data, y_data_2, label='Attack')
    # plt.plot(x_data, y_data_3, label='Defense')
    # plt.plot(x_data, y_data_4, label='SP Attack')
    # plt.plot(x_data, y_data_5, label='SP Defense')
    # plt.plot(x_data, y_data_6, label='Speed')

    plt.legend()

    plt.xlabel('Skill Points')
    plt.ylabel('Pokemon')
    plt.title('Pokemons power')

    plt.show()


def part6():
    print_razd(6)
    # T =TSNE(n_components=)
    global data
    df = data
    df.shape
    non_numeric = ['name', 'type1', 'type2', 'legendary']
    df_numeric = df.drop(non_numeric, axis=1)
    df_numeric.shape
    m = TSNE(learning_rate=50)
    tsne_features = m.fit_transform(df_numeric)
    tsne_features[1:4, :]
    df['total_power'] = tsne_features[:, 0]
    df['generation'] = tsne_features[:, 1]
    sns.scatterplot(x="total_power", y="generation", data=df, hue=data['type1'])
    plt.show()


def part6_v2():
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))


def part7():
    print_razd(7)
    global data
    df = data
    df.shape
    non_numeric = ['name', 'type1', 'type2', 'legendary']
    df_numeric = df.drop(non_numeric, axis=1)
    df_numeric.shape
    scaler = preprocessing.MinMaxScaler()
    df_numeric = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)
    n_n = (10, 100, 1000)
    m_d = (0.05, 0.2, 0.9)
    um = dict()
    for n_neighbors in n_n:
        for min_dist in m_d:
            umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
            start_time = time.time()
            x_embedded = umap.fit_transform(df_numeric)
            end_time = time.time()
            time_took = end_time - start_time
            plt.figure(figsize=(8, 6))
            plt.scatter(
                x_embedded[:, 0],
                x_embedded[:, 1],
                cmap=plt.colormaps.get_cmap(cmap="jet"),
                c=data['generation'].astype(int)

            )
            plt.colorbar(ticks=range(10))
            plt.title(f"UMAP (with n_neighbors {n_neighbors}, min_dist {min_dist})")
            print(f"UMAP – time (with n_neighbors {n_neighbors}, min_dist {min_dist}): {time_took:.2f} seconds")
            plt.show()


import time
from umap import UMAP


def part7_2(x, y, n_neighbors_values, min_dist_values):
    print_razd(7)
    for n_neighbors in n_neighbors_values:
        for min_dist in min_dist_values:
            umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
            start_time = time.time()
            x_embedded = umap.fit_transform(x)
            end_time = time.time()
            time_took = end_time - start_time
            plt.figure(figsize=(8, 6))
            plt.scatter(
                x_embedded[:, 0],
                x_embedded[:, 1],
                c=y,
                cmap=plt.colormaps.get_cmap(cmap="jet"),
            )
            plt.colorbar(ticks=range(10))
            plt.title(f"UMAP (with n_neighbors {n_neighbors}, min_dist {min_dist})")
            print(f"UMAP – time (with n_neighbors {n_neighbors}, min_dist {min_dist}): {time_took:.2f} seconds")
            plt.show()


if __name__ == '__main__':
    # part1()
    # part2()
    # part3()
    # part4()
    part5()
    # part6()
    #part7()
