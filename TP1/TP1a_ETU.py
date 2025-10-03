

"""
Created on Thu Mar 27 10:37:24 2025

@author: cathe
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode

def visualize_classifier(model, X, y):
    ax = plt.gca()
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=1, cmap='rainbow',
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Create a color plot with the results
    n_classes = len(np.unique(y))
    plt.scatter(xx.ravel(), yy.ravel(), c=Z, s=0.1, cmap='rainbow');
    ax.set(xlim=xlim, ylim=ylim)
    plt.show()

data = np.load("TP1a.npz")
X_train, y_train, X_test, y_test = (data[key] for key in ["X_train", "y_train", "X_test", "y_test"])

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=1, cmap='rainbow');
plt.show()


def predict_multiple(Nrun:int = 30, proportion:float = 0.6, model_type:str = 'knn', k = 1):
    preds_all = np.zeros((Nrun, len(X_test)), dtype=int)  # allocation une seule fois
    
    for run in range(Nrun):
        random_state = 61 + run
        idx = np.random.RandomState(random_state).choice(
            len(X_train), size=int(proportion*len(X_train)), replace=False
        )
        X_sub, y_sub = X_train[idx], y_train[idx].ravel()

        if model_type == 'knn':
            model = KNeighborsClassifier(n_neighbors=k)
        elif model_type == 'tree':
            model = DecisionTreeClassifier(criterion='entropy', max_depth=k, random_state=8061)
        elif model_type == 'rf':
            model = RandomForestClassifier(criterion='entropy', n_estimators=k, random_state=61, max_depth=9)
        else:
            print("Unsupported model type")
            return

        model.fit(X_sub, y_sub)

        preds_all[run] = model.predict(X_test)

    return preds_all


def get_results(Nrun, preds):
    biais_list = []
    variance_list = []

    for i in range(len(X_test)):
        y_true = y_test[i]

        pred_mode, count = mode(preds[:, i], keepdims=False)
        pred_mode = pred_mode.item()
        count = count.item()

        biais = 0 if pred_mode == y_true else 1
        variance = 1 - count / Nrun

        biais_list.append(biais)
        variance_list.append(variance)

    biais_moy = np.mean(biais_list)
    variance_moy = np.mean(variance_list)
    return biais_moy, variance_moy


