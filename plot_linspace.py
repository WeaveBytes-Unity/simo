from sklearn.svm import SVC
import numpy as np
import pandas as pd # to import csv and for data manipulation
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split # to split the data

iris = datasets.load_iris()
# Select 2 features / variable for the 2D plot that we are going to create.
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

# filename = "datasets/data.csv" 

# contraFrame = pd.read_csv(filename,header=None)
# colnames =['wife_age','wife_education', 'husband_education', 'children', 'wife_relogion', 
# 	'wife_working','husband_occupation','standard_of_living', 'media_exposure','contraceptive_method']
# data = pd.read_csv(filename, names = colnames, header=None)

# X = data.ix[:,data.columns != "contraceptive_method"]
# Y = data.ix[:,data.columns == "contraceptive_method"]

# x_train,x_test,y_train,y_test = train_test_split(features, labels, test_size=0.2, random_state=12)

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

model = svm.SVC(kernel='linear')
clf = model.fit(X, Y)

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVC ')
# Set-up grid for plotting.
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('y label here')
ax.set_xlabel('x label here')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()