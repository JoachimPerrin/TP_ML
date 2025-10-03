

import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from sklearn import svm
from random import gauss

def aff_donnees(X,y,bornex,borney,s):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=s, cmap='winter');
    plt.xlim(bornex);
    plt.ylim(borney);

def aff_plan(w,b,bornex, color = 'r', linestyle='-'):
    x=np.linspace(bornex[0],bornex[1],10)
    y=-(w[0]*x+b)/w[1]
    plt.plot(x,y,color=color, linestyle=linestyle)

def Resoud_primal(X,y):
    N = X.shape[0]
    n = X.shape[1]
    q = cvxopt.matrix(np.zeros((n+1, 1)))
    h = -np.ones((N,1))
    h = cvxopt.matrix(h)

    P1=np.concatenate((np.zeros((1,1)),np.zeros((1,n))),axis=1)
    P2=np.concatenate((np.zeros((n,1)),np.eye(n)),axis=1)
    P=np.concatenate((P1,P2),axis=0)
    P=cvxopt.matrix(P)
    
    for i in range(N):
        g=np.concatenate((np.reshape(-y[i],(1,1)), np.reshape(-y[i]*X[i][:],(1,2))),axis=1)
        if i==0:
            G=g
        else:
            G=np.concatenate((G, g), axis=0)

    G=cvxopt.matrix(G+0.)
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q, G, h)
    z=sol['x']
    w = np.array(z[1:n+1])
    b = z[0]
    return w,b

def Resoud_primal_souple(X,y,C):
    N = X.shape[0]
    n = X.shape[1]
    q = cvxopt.matrix(np.concatenate((np.zeros((n+1, 1)), C*np.ones((N, 1))), axis=0))
    h = cvxopt.matrix(np.concatenate((-np.ones((N,1)), np.zeros((N,1))), axis=0))

    P1=np.concatenate((np.zeros((1,1)),np.zeros((1,n)), np.zeros((1,N))),axis=1)
    P2=np.concatenate((np.zeros((n,1)),np.eye(n), np.zeros((n,N))),axis=1)
    P3=np.concatenate((np.zeros((N,1)), np.zeros((N,n)), np.zeros((N,N))),axis=1)
    P=np.concatenate((P1,P2, P3),axis=0)
    P=cvxopt.matrix(P)

    for i in range(N):
        g=np.concatenate((np.reshape(-y[i],(1,1)), np.reshape(-y[i]*X[i][:],(1,2))),axis=1)
        if i==0:
            G=g
        else:
            G=np.concatenate((G, g), axis=0)

    G = np.concatenate((G, -np.eye(N)), axis = 1)
    G = np.concatenate((G, np.concatenate((np.zeros((N,n+1)), -np.eye(N)), axis=1)), axis = 0)

    G=cvxopt.matrix(G+0.)
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q, G, h)
    z=sol['x']
    b = z[0]
    w = np.array(z[1:n+1])
    xi = np.array(z[n+1:N+n+1])

    return w,b


def aff_frontiere(X,y,bornex,borney,model):
    aff_donnees(X,y,bornex,borney,50)
    xx, yy = np.meshgrid(np.linspace(bornex[0], bornex[1],50), np.linspace(borney[0], borney[1],50))
    xy = np.concatenate((np.reshape(xx,(xx.shape[0]*xx.shape[1],1)),np.reshape(yy,(yy.shape[0]*yy.shape[1],1))),axis=1)
    P = model.predict(xy)
    aff_donnees(xy,P,bornex,borney,1)





