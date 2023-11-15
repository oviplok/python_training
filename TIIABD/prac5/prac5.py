import pandas as pd
import matplotlib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

matplotlib.use('TkAgg')  # Use the TkAgg backend
import matplotlib.pyplot as plt


def part1():
    data = pd.read_csv('spotify_songs.csv')

    # Подсчет количества экземпляров каждого класса
    class_counts = data['playlist_genre'].value_counts()

    # Построение гистограммы
    plt.bar(class_counts.index, class_counts.values)
    plt.xlabel('Жанр')
    plt.ylabel('Количесвто плейлистов')
    plt.title('Баланс классов')
    plt.show()


def part2():
    data = pd.read_csv('spotify_songs.csv')

    # Определение признаков (X) и целевой переменной (y)
    X = data.drop('playlist_genre', axis=1)  # замените 'target_column' на название целевой переменной
    y = data['playlist_genre']  # замените 'target_column' на название целевой переменной

    # Разделение выборки на тренировочную и тестовую
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)  # тут 0.2 - доля тестовой выборки, random_state для воспроизводимости результатов


def part3():
    # X = df[['speechiness', 'acousticness', 'instrumentalness']]  # Замените на реальные названия признаков
    # y = df['tempo']
    df = pd.read_csv('spotify_songs.csv')

    # Разделение данных на тренировочные и тестовые
    X = df[['speechiness', 'acousticness', 'instrumentalness']]  # Замените на реальные названия признаков
    y = df['track_popularity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Инициализация моделей
    logreg_model = LogisticRegression()
    svm_model = SVC()
    knn_model = KNeighborsClassifier()

    # Обучение моделей на тренировочных данных
    logreg_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)

    # Получение предсказаний моделей на тестовых данных
    logreg_pred = logreg_model.predict(X_test)
    svm_pred = svm_model.predict(X_test)
    knn_pred = knn_model.predict(X_test)

    # Построение матрицы ошибок для каждой модели
    logreg_cm = confusion_matrix(y_test, logreg_pred)
    svm_cm = confusion_matrix(y_test, svm_pred)
    knn_cm = confusion_matrix(y_test, knn_pred)

    print("Confusion Matrix for Logistic Regression:")
    print(logreg_cm)
    print("\nConfusion Matrix for SVM:")
    print(svm_cm)
    print("\nConfusion Matrix for KNN:")
    print(knn_cm)

    print("PART 5 ------------------------------")

    # Печать отчета для каждой модели
    print("Classification Report for Logistic Regression:")
    print(classification_report(y_test, logreg_pred))
    print("\nClassification Report for SVM:")
    print(classification_report(y_test, svm_pred))
    print("\nClassification Report for KNN:")
    print(classification_report(y_test, knn_pred))


if __name__ == '__main__':
    part3()
