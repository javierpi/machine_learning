import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random


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


df = pd.read_csv("breast-cancer-wisconsin.data")
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
# print(df.head())
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)
print(20*'#')
print(full_data[:5])
