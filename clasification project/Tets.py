import heapq
import numpy as np
import pandas as pd
import matplotlib as plt
import math
import sklearn
from sklearn.cluster import KMeans


# X = np.array([[1,2,1,2],
#               [1,3,1,3],
#               [3,1,1,1]])
# print(X)
# weight = [1,2,3,4]
# X =np.array([x*weight for x in X])
# print(X)

# gruop_map = np.zeros((4, 10, 8))
# print(gruop_map)
# n = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
# print(n)
# index = np.random.choice(n, size=1)
# a = n[index]
# print(a)
# # n = np.delete(n, index)
# # print(n)
#
# X = list([np.arange(0, 5)])*4
# print(X)
# index = np.random.choice(X[2],size=1)
# a = X[2][index]
# X[2] = np.delete(X[2],index)
# print(a)
# print(X0
#
#
# # X = np.random.random((4,8))
# # Y = np.random.random((4,8))
# X = np.array([[2, 0],
#               [3, 4]])
# Y = np.array([[2, 0],
#               [1, 3]])
# print(cost_function_2(X, Y))

choice_list = list(np.arange(0,8))
a = list(np.random.choice(choice_list,size=2,replace=False))
print(a)

