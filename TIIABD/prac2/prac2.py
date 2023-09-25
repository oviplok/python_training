import matplotlib
import pandas as pd
from sklearn.datasets import fetch_openml
import csv
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

matplotlib.use('TkAgg')

count = 0

# Загрузка данных из файла CSV
data = pd.read_csv('pokemon.csv')


def print_razd():
    global count
    count = count + 1
    print("\npart " + str(count) + " //////////////////////////////////////////////////////////////////////")


def part1():
    print_razd()
    # Описание найденных данных
    description = data.describe()
    # Вывод описания данных
    print(description)


def part2():
    print_razd()
    # Описание c помощью info и head
    info = data.info
    head = data.head()
    # Вывод описания данных
    print(info)
    print()
    print(head)


def part3():
    print_razd()
    global data
    x_data = data['type1']
    y_data = data['generation']
    # color_data = data['признак']

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
    print_razd()
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
    print_razd()
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
    plt.plot(x_data, y_data, color='crimson', linewidth=2, label='Total Power')
    # plt.plot(x_data, y_data_1, label='HP')
    # plt.plot(x_data, y_data_2, label='Attack')
    # plt.plot(x_data, y_data_3, label='Defense')
    # plt.plot(x_data, y_data_4, label='SP Attack')
    # plt.plot(x_data, y_data_5, label='SP Defense')
    # plt.plot(x_data, y_data_6, label='Speed')

    # Добавление легенды
    plt.legend()
    # Настройка осей и заголовка
    plt.xlabel('Skill Points')
    plt.ylabel('Pokemon')
    plt.title('Pokemons power')

    # Отображение графика
    plt.show()


def part6():
    print_razd()
    global data
    df = data
    df.shape
    non_numeric = ['name', 'type1', 'type2', 'legendary']
    df_numeric = df.drop(non_numeric, axis=1)
    df_numeric.shape
    m = TSNE(learning_rate=50)
    tsne_features = m.fit_transform(df_numeric)
    tsne_features[1:4,:]


def part7():
    pass


if __name__ == '__main__':
    # part1()
    # part2()
    # part3()
    # part4()
    # part5()
    part6()
    part7()
