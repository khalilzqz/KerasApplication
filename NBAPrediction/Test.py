# -*- encoding:utf-8 -*-
import numpy as np

a = np.array([[1, 2],
              [3, 4],
              [5, 6]])
b = np.array([[5, 6, 7]])

# print(np.concatenate((a, b), axis=0))

print(np.concatenate((a, b.T), axis=1))
