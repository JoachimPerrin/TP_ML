

"""
Created on Thu Mar 27 10:37:24 2025

@author: cathe
"""
import numpy as np
import matplotlib.pyplot as plt



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