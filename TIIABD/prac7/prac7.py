import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import time

df = pd.read_csv('spotify_songs.csv')
df = df.drop(df[(df.playlist_genre == 'rap') | (df.playlist_genre == 'pop') | (df.playlist_genre == 'r&b')].index)

X = df[['speechiness', 'acousticness', 'instrumentalness']]
y = df['mode']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

start_time_bagging = time.time()
# Инициализация баггинга
bagging_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
# Обучение баггинга
bagging_model.fit(X_train, y_train)
# Прогнозирование
bagging_predictions = bagging_model.predict(X_test)
bagging_accuracy = accuracy_score(y_test, bagging_predictions)
end_time_bagging = time.time()
bagging_time = end_time_bagging - start_time_bagging
# Оценка качества баггинга
bagging_mse = mean_squared_error(y_test, bagging_predictions)

start_time_boosting = time.time()
# Инициализация бустинга
boosting_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42)
# Обучение бустинга
boosting_model.fit(X_train, y_train)
# Прогнозирование
boosting_predictions = boosting_model.predict(X_test)
boosting_accuracy = accuracy_score(y_test, boosting_predictions)
end_time_boosting = time.time()
boosting_time = end_time_boosting - start_time_boosting
# Оценка качества бустинга
boosting_mse = mean_squared_error(y_test, boosting_predictions)

# Сравнение результатов баггинга и бустинга
print("Accuracy of Bagging:", bagging_accuracy)
print("Time taken for Bagging:", bagging_time)
print("Bagging MSE:", bagging_mse)
print("")
print("Accuracy of Boosting:", boosting_accuracy)
print("Time taken for Boosting:", boosting_time)
print("Boosting MSE:", boosting_mse)

# plt.title("Boosting")
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.scatter(X, boosting_predictions, c='green', s=16, zorder=2)
# plt.plot(X, y, '--', color='black', lw=1.5)
# plt.show()
#
# plt.title("Bagging")
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.scatter(X, bagging_predictions, c='green', s=16, zorder=2)
# plt.plot(X, y, '--', color='black', lw=1.5)
# plt.show()


