import pandas as pd
import matplotlib
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px

matplotlib.use('TkAgg')  # Use the TkAgg backend
import matplotlib.pyplot as plt

df = pd.read_csv('spotify_songs.csv')
df = df.drop(df[(df.playlist_genre == 'rap') | (df.playlist_genre == 'pop') | (df.playlist_genre == 'r&b')].index)


def part1():
    global df
    # Подсчет количества экземпляров каждого класса
    class_counts = df['playlist_genre'].value_counts()

    # Построение гистограммы
    plt.bar(class_counts.index, class_counts.values)
    plt.xlabel('Жанр')
    plt.ylabel('Количесвто плейлистов')
    plt.title('Баланс классов')
    plt.show()


def part2():
    global df

    # Определение признаков (X) и целевой переменной (y)
    X = df.drop('playlist_genre', axis=1)
    y = df['playlist_genre']

    # Разделение выборки на тренировочную и тестовую
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)


def part3():
    global df

    # Разделение данных на тренировочные и тестовые
    X = df[['speechiness', 'acousticness', 'instrumentalness']]
    y = df['playlist_genre']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f"Train subset size – {len(X_train)}")
    print(f"Testing subset size – {len(X_test)}")

    # Logistic Regression
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(X_train, y_train)
    y_pred_logistic = logistic_regression_model.predict(X_test)
    conf_matrix_logistic = confusion_matrix(y_test, y_pred_logistic)

    # SVM (Support Vector Machine)
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

    # K-Nearest Neighbors
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)

    # Print confusion matrices
    print("Confusion Matrix - Logistic Regression:")
    print(conf_matrix_logistic)
    fig = px.imshow(conf_matrix_logistic, text_auto=True)
    fig.update_layout(xaxis_title="Target", yaxis_title="Prediction")
    fig.show()

    print("\nConfusion Matrix - SVM:")
    print(conf_matrix_svm)
    fig = px.imshow(conf_matrix_svm, text_auto=True)
    fig.update_layout(xaxis_title="Target", yaxis_title="Prediction")
    fig.show()

    print("\nConfusion Matrix - K-Nearest Neighbors:")
    print(conf_matrix_knn)
    fig = px.imshow(conf_matrix_knn, text_auto=True)
    fig.update_layout(xaxis_title="Target", yaxis_title="Prediction")
    fig.show()

    print("PART 5 ------------------------------")

    # Печать отчета для каждой модели
    precision = metrics.precision_score(y_test, y_pred_svm, zero_division=1, average='weighted')
    f1 = metrics.f1_score(y_test, y_pred_svm, zero_division='warn', average='weighted')

    # Печать отчета для каждой модели
    print("Classification Report - Logistic Regression:")
    print(classification_report(y_test, y_pred_logistic))

    print("\nClassification Report - SVM:")
    print(classification_report(y_test, y_pred_svm))
    print("\nClassification Report - K-Nearest Neighbors:")
    print(classification_report(y_test, y_pred_knn))


if __name__ == '__main__':
    part3()
