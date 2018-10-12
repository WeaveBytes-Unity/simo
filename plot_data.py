from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # to import csv and for data manipulation

from sklearn import svm, datasets
# iris = datasets.load_iris()
# # Select 2 features / variable for the 2D plot that we are going to create.
# X = iris.data[:, :2]  # we only take the first two features.
# y = iris.target

# filename = "datasets/data.csv" 

# contraFrame = pd.read_csv(filename,header=None)
# colnames =['wife_age','wife_education', 'husband_education', 'children', 'wife_relogion', 
# 	'wife_working','husband_occupation','standard_of_living', 'media_exposure','contraceptive_method']
# data = pd.read_csv(filename, names = colnames, header=None)

# X = data.iloc[:, :2] #data.columns != "contraceptive_method"]
# y = data.iloc[:,data.columns == "contraceptive_method"].values.ravel()

# print(X.shape[0])
# print(X.shape[1])
# print(y)
# print(y.shape)

# def make_meshgrid(x, y, h=.02):
#     x_min, x_max = x.min() - 1, x.max() + 1
#     # print(x_max)
#     # print(x_min)
#     y_min, y_max = y.min() - 1, y.max() + 1
#     # print(y_max)
#     # print(y_min)

#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     return xx, yy

# def plot_contours(ax, clf, xx, yy, **params):
# 	# Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])	
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     out = ax.contourf(xx, yy, Z, **params)
#     return out

# model = svm.SVC(kernel='linear', C=0.001)
# clf = model.fit(X, y)

# fig, ax = plt.subplots()
# # title for the plots
# title = ('Decision surface of linear SVC ')

# # Set-up grid for plotting.
# # X0, X1 = Xreduced[:, 0], Xreduced[:, 1]
# X0, X1 = X.values[:, 0], X.values[:, 1]
# xx, yy = make_meshgrid(X0, X1)
# # print("xx => ",xx)
# # print("yy => ",yy)

# # ax.scatter(xx, yy, color="blue", s=10, label="Healthy")
# # plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
# ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
# ax.set_ylabel('y label here')
# ax.set_xlabel('x label here')
# ax.set_xticks(())
# ax.set_yticks(())
# ax.set_title(title)
# ax.legend()
# plt.show()

#========================================================
from sklearn.datasets.samples_generator import make_blobs

fig, axis = plt.subplots()
X, y = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=2.7)
print(X)
print("X type ", type(X))
print(y)
print("y type ", type(y))

axis.scatter(X[y == 0, 0], X[y == 0, 1], color="red", s=10, label="Disease")
axis.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", s=10, label="Healthy")
axis.scatter(X[y == 2, 0], X[y == 2, 1], color="green", s=10, label="Fit")


clf = svm.SVC(kernel='linear' , C=1.0).fit(X, y)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 7)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the line, the points, and the nearest vectors to the plane
axis.plot(xx, yy, 'k-', color="black", label="Model")

w = clf.coef_[1]
a = -w[0] / w[1]
xx = np.linspace(-5, 7)
yy = a * xx - (clf.intercept_[1]) / w[1]

axis.plot(xx, yy, 'k-', color="black", label="Model")


w = clf.coef_[2]
a = -w[0] / w[1]
xx = np.linspace(-5, 7)
yy = a * xx - (clf.intercept_[2]) / w[1]

axis.plot(xx, yy, 'k-', color="black", label="Model")

axis.tick_params(labelbottom='off', labelleft='off')
axis.set_xlabel("Gene 1")
axis.set_ylabel("Gene 2")
axis.legend() 
plt.show()

