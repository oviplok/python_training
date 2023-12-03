import pandas as pd
import matplotlib
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, silhouette_score
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.cluster import AgglomerativeClustering
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go

matplotlib.use('TkAgg')  # Use the TkAgg backend
import matplotlib.pyplot as plt

# df = pd.read_csv('spotify_songs.csv')
df = (pd.read_csv('spotify_songs.csv',
                  usecols=['speechiness',
                           'acousticness',
                           'instrumentalness']).dropna().head(5000))


# df = df.drop(df[(df.playlist_genre == 'rap') | (df.playlist_genre == 'pop') | (df.playlist_genre == 'r&b')].index)

def part2():
    models = []
    scores_inertia = []
    scores_silhouette = []

    UPPER_K_BOUNDARY = 10
    for i in range(2, UPPER_K_BOUNDARY):
        model = KMeans(n_clusters=i, n_init=10, init="k-means++").fit(df)
        models.append(model)
        scores_inertia.append(model.inertia_)
        scores_silhouette.append(silhouette_score(df, model.labels_))

    plt.grid()
    plt.plot(np.arange(2, UPPER_K_BOUNDARY), scores_inertia, marker="o")
    plt.show()

    plt.grid()
    plt.plot(np.arange(2, UPPER_K_BOUNDARY), scores_silhouette, marker="o")
    plt.show()


def part2_viz():
    CLUSTERS_CHOSEN = 4
    chosen_model_kmeans = KMeans(n_clusters=CLUSTERS_CHOSEN, n_init=10, init="k-means++").fit(df)
    print(chosen_model_kmeans.cluster_centers_)
    df_kmeans_copy = df.copy()
    df_kmeans_copy["Cluster"] = chosen_model_kmeans.labels_
    print(df_kmeans_copy["Cluster"].value_counts())
    fig = go.Figure(data=[go.Scatter3d(x=df_kmeans_copy["speechiness"]
                                       , y=df_kmeans_copy["acousticness"]
                                       , z=df_kmeans_copy["instrumentalness"]
                                       , mode="markers"
                                       , marker_color=df_kmeans_copy["Cluster"]
                                       , marker_size=2,
                                       )])
    fig.show()


def part3():
    CLUSTERS_CHOSEN = 4
    chosen_model_agglomerative = AgglomerativeClustering(CLUSTERS_CHOSEN, compute_distances=True).fit(df)
    df_agglomerative_copy = df.copy()
    df_agglomerative_copy["Cluster"] = chosen_model_agglomerative.labels_
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=df_agglomerative_copy["speechiness"],
                y=df_agglomerative_copy["acousticness"],
                z=df_agglomerative_copy["instrumentalness"],
                mode="markers",
                marker_color=df_agglomerative_copy["Cluster"],
                marker_size=2, )])

    fig.update_layout(
        autosize=False,
        width=1000,
        height=700,
        margin=dict(l=10, r=10, b=0, t=20, pad=4),
    )
    fig.show()


def part4():
    chosen_model_dbscan = DBSCAN(eps=2)
    df_dbscan_copy = df.copy()
    df_dbscan_copy["Cluster"] = chosen_model_dbscan.fit(df).labels_
    print("Clusters:", df_dbscan_copy["Cluster"].unique())
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=df_dbscan_copy["speechiness"],
                y=df_dbscan_copy["acousticness"],
                z=df_dbscan_copy["instrumentalness"],
                mode="markers",
                marker_color=df_dbscan_copy["Cluster"],
                marker_size=2,
            ),
        ]
    )
    fig.update_layout(
        autosize=False,
        width=1000,
        height=700,
        margin=dict(l=10, r=10, b=0, t=20, pad=4),
    )
    fig.show()


if __name__ == '__main__':
    part2()
    part2_viz()
    part3()
    part4()
