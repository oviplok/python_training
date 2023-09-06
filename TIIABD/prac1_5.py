import pandas as pd
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)

if __name__ == '__main__':
    # Преобразование датасета в объект DataFrame из pandas
    df = pd.DataFrame(data.data, columns=data.feature_names)

    print(df.info())

    print("\n/////////////////////////////////////////////////////////\n")

    print(df.isna().sum())

    print("\n/////////////////////////////////////////////////////////\n")

    print(df.loc[(df['AveBedrms'] > 50) & (df['Population'] > 2500)])

    print("\n/////////////////////////////////////////////////////////\n")

    # Получение массива значений медианной стоимости домов
    median_house_prices = data.target

    # Нахождение максимального и минимального значений
    max_price = max(median_house_prices)
    min_price = min(median_house_prices)

    # Вывод результатов
    print("max:", max_price)
    print("min:", min_price)
    print("\n/////////////////////////////////////////////////////////\n")
    # Вывод названия признака и его среднего значения
    mean_values = df.mean()
    for feature, mean_value in mean_values.items():
        print("Название признака:", feature)
        print("Среднее значение:", mean_value)
        print()
