import numpy as np
import pandas as pd
import heapq
from sklearn.cluster import KMeans
import sys


def compute_cost(t: int, k: int, data: pd.DataFrame, gruop_map, result, structure):
    cost = [0] * t
    for i in range(0, t):
        X_data = np.zeros((k, t))
        for j in range(0, k):
            X_data[j] = np.array(data.iloc[int(gruop_map[j, result[i, j]]), :-1])
        cost[i] = cost_function(X_data, structure)
    return cost


def find_largest_n(X: list, n):
    a = heapq.nlargest(n, enumerate(X), key=lambda x: x[1])
    index, vals = zip(*a)
    return list(index)


def cost_function(X: np.ndarray, Y: np.ndarray):
    k = X.shape[0]
    t = X.shape[1]
    cost = 0
    for i in range(0, t):
        temp = 0
        for j in range(0, k):
            temp += X[j, i] - Y[j, i]
        cost += abs(temp)
    return cost


def add_weights(X: np.array, weight: list):
    X = np.array([x * weight for x in X])
    return X


def regularit(df: pd.DataFrame):
    newDataFrame = pd.DataFrame(index=df.index)
    columns = df.columns.tolist()
    for c in columns:
        d = df[c]
        MAX = d.max()
        MIN = d.min()
        newDataFrame[c] = ((d - MIN) / (MAX - MIN)).tolist()
    return newDataFrame


def do_k_means(X: np.ndarray, k):
    kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=500).fit(X)
    return kmeans.inertia_, kmeans.labels_


# read student data and expected description of group
data_path = input("please enter the data path: ")
data = pd.read_csv(data_path)

# design expected group
is_customized = input("do you want to describe what your expect group look like? [y/n]")
while is_customized != 'n' and is_customized != 'y' and is_customized != 'Y' and is_customized != 'N':
    is_customized = input("do you want to describe what your expect group look like? [y/n]")
if is_customized == 'y' or is_customized == 'Y':
    structure_path = input("please enter the description file path: ")
    structure = pd.read_csv(structure_path)
    structure = np.array(structure.iloc[:, :])
    k = structure.shape[0]
else:
    k = int(input("please input the size of each group: "))
    structure = np.zeros((k, int(data.shape[0] / k)))

# design k-means features weight
is_weighted = input("do you want to describe the weight of each features? [y/n]")
while is_weighted != 'n' and is_weighted != 'y' and is_weighted != 'Y' and is_weighted != 'N':
    is_weighted = input("do you want to describe the weight of each features? [y/n]")
if is_weighted == 'y' or is_weighted == 'Y':
    weight_path = input("please enter the weight file path: ")
    weight = pd.read_csv(weight_path)
    weight = list(np.array(weight.iloc[:, :]).flatten())
else:
    weight = [1] * data.shape[1]

# regularit the data
data = regularit(data)
X = np.array(data.iloc[:, :])

# add the influence of weights

X = add_weights(X, weight)
structure = add_weights(structure, weight)

# do k-means and get the labels
counter = 0
mincost = sys.maxsize
perfect_labels = []
while counter <= 5:
    cost, labels = do_k_means(X, k)
    if cost < mincost:
        mincost = cost
        perfect_labels = labels
    counter += 1
data['label'] = labels
print(data)

# get the labeled data
t = int(data.shape[0] / k)
gruop_map = np.zeros((k, t))
index_list = [0] * k
for index, row in data.iterrows():
    y = int(row['label'])
    z = index_list[y]
    if (z >= t):
        y = 0
    z = index_list[y]
    while (z >= t):
        y += 1
        z = index_list[y]
    gruop_map[y, index_list[y]] = index
    index_list[y] += 1

print(gruop_map)

# get the random matching
result = np.zeros((t, k), dtype=int)
random_map = list([np.arange(0, t)]) * k
for i in range(0, t):
    for j in range(0, k):
        a = np.random.choice(random_map[j], size=1)
        index = 0
        for m in range(0, len(random_map[j])):
            if (random_map[j][m] == a):
                index = m
                break
        result[i, j] = a
        random_map[j] = np.delete(random_map[j], index)

# get the initial lost function
cost = [0] * t
for i in range(0, t):
    X_data = np.zeros((k, t))
    for j in range(0, k):
        X_data[j] = np.array(data.iloc[int(gruop_map[j, result[i, j]]), :-1])
    cost[i] = cost_function(X_data, structure)

print(cost)
print("sum of cost" + str(sum(cost)))
print(result)

# do the minimize alg
# iterations
for i in range(0, 100):
    # find two largest cost group
    # two_largest_group = find_largest_n(cost, 2)
    choice_list = list(np.arange(0, t))
    two_largest_group = list(np.random.choice(choice_list, size=2, replace=False))
    group_A = result[two_largest_group[0]]
    group_B = result[two_largest_group[1]]
    # print(group_A)
    # print(group_B)

    # decide each cost when exchange one member of the choisen 2 group
    changed_cost = [0] * k
    for j in range(0, k):
        # do exchange
        tempA = group_A.copy()
        tempB = group_B.copy()
        temp = tempA[j]
        tempA[j] = tempB[j]
        tempB[j] = temp

        # add tempA cost
        X_data = np.zeros((k, t))
        for q in range(0, k):
            X_data[q] = np.array(data.iloc[int(gruop_map[q, tempA[q]]), :-1])
        changed_cost[j] += cost_function(X_data, structure)
        # add tempB cost
        X_data = np.zeros((k, t))
        for q in range(0, k):
            X_data[q] = np.array(data.iloc[int(gruop_map[q, tempB[q]]), :-1])
        changed_cost[j] += cost_function(X_data, structure)
    # print(changed_cost)

    # find the best variation
    index_of_smallest = 0
    smallest = changed_cost[index_of_smallest]
    for j in range(0, k):
        if changed_cost[j] < smallest:
            index_of_smallest = j
            smallest = changed_cost[j]
    # print(index_of_smallest)

    # judge if the best variation is better than parents
    best_variaiton_cost = changed_cost[index_of_smallest]
    parents_cost = cost[two_largest_group[0]] + cost[two_largest_group[1]]
    # print(best_variaiton_cost)
    # print(parents_cost)

    if best_variaiton_cost < parents_cost:
        # do the variation
        temp_2 = result[two_largest_group[0], index_of_smallest]
        result[two_largest_group[0], index_of_smallest] = result[two_largest_group[1], index_of_smallest]
        result[two_largest_group[1], index_of_smallest] = temp_2
        # compute each itr's cost
        cost = compute_cost(t, k, data, gruop_map, result, structure)
        print(str(sum(cost)))
    else:
        pass

print(str(sum(cost)))
final_group = np.zeros((t,k),dtype=int)
for i in range (0,t):
    for j in range(0,k):
        final_group[i,j] = gruop_map[j,result[i,j]]
# print(gruop_map)
# print(result)
print(final_group)