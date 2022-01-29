import numpy as np
import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN
from sklearn import svm
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from data_utils import *
from collections import Counter


def select_MinPts(data, k):
    k_dist = []
    for i in range(data.shape[0]):
        dist = (((data[i] - data)**2).sum(axis=1)**0.5)
        dist.sort()
        k_dist.append(dist[k])
    return np.array(k_dist)


def clean_i(data, eps_index, features):
    k = features * 2 - 1
    k_dist = select_MinPts(data, k)
    k_dist.sort()
    # x = np.arange(k_dist.shape[0])
    # y = k_dist[::-1]
    # plt.plot(x, y)
    # plt.show()
    eps = k_dist[::-1][eps_index]
    dbscan_model = DBSCAN(eps=eps, min_samples=k+1)
    label = dbscan_model.fit_predict(data)
    temp = Counter(label)
    del temp[-1]
    # print(temp)
    max_len = 0
    ret_index = -1
    for i in range(len(temp)):
        len_i = temp[i]
        if len_i > max_len:
            max_len = len_i
            ret_index = i

    ret = np.argwhere(label == ret_index)
    new_data = data[ret].reshape(-1, features).tolist()
    return new_data

    # new_data = data
    # k_means = KMeans(n_clusters=2, random_state=217)
    # k_means.fit(new_data)
    # r = Counter(k_means.labels_)
    # print(r)
    # label2 = np.array(k_means.labels_)
    # index = np.argwhere(label2 == 0)
    # if r[0] < r[1]:
    #     index = np.argwhere(label2 == 1)
    #
    # return new_data[index].reshape(-1, 16).tolist()





