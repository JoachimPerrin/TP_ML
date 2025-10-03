# -*- coding: utf-8 -*-
"""
Created on Fri May  9 20:01:32 2025

@author: cathe
"""

import numpy as np
import matplotlib.pyplot as plt

# data = np.load("TP5.npz")
# X_train, y_train, X_test, y_test = (data[key] for key in ["X_train", "y_train", "X_test", "y_test"])
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=1, cmap='rainbow');
# plt.show()
# plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=1, cmap='rainbow');
# plt.show()

data = np.load("TP1a.npz")
X_train, y_train, X_test, y_test = (data[key] for key in ["X_train", "y_train", "X_test", "y_test"])
print(X_train.shape, X_test.shape) # renvoie (16640, 2) (4160, 2)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=1, cmap='rainbow');
plt.show()