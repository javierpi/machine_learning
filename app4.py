import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter


########################
# Euclidean Distance
#     _________________
#    /  n
# = V  ----
#      \            2
#       /  ( a - p )
#      /      i   i
#      ----
#      i = 1
##########################
# q = (1,3)
# P = (2,5)
# Res = sqrt( ((q[1]- p[1])**2  + ((q[2]-p[2])**2) )

def tipo1():
    plot1 = [1, 3]
    plot2 = [2, 5]

    euclidean_distance = sqrt(((plot1[0] - plot2[0])**2) + ((plot1[1] - plot2[1])**2))

    print(euclidean_distance)


style.use('ggplot')
dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]


# # for i in dataset:
# #     for ii in dataset[i]:
# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1])
# plt.show()


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value greater than total voting groups!')

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    return(vote_result)


value = k_nearest_neighbors(dataset, new_features, 2)
print(value)
[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], color=value)
plt.show()
