import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd # to import csv and for data manipulation


filename = "datasets/data.csv" 

contraFrame = pd.read_csv(filename,header=None)
colnames =['wife_age','wife_education', 'husband_education', 'children', 'wife_relogion', 
    'wife_working','husband_occupation','standard_of_living', 'media_exposure','contraceptive_method']
data = pd.read_csv(filename, names = colnames, header=None)


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    print(x_max)
    print(x_min)
    y_min, y_max = y.min() - 1, y.max() + 1
    print(y_max)
    print(y_min)

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_decision_function(classifier, sample_weight, axis, title):
    # plot the decision function
    # xx, yy = make_meshgrid(X.values[:, 0], X.values[:, 1])
    xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plot the line, the points, and the nearest vectors to the plane
    axis.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.bone)
    axis.scatter(X.values[:, 0], X.values[:, 1], c=y, s=100 * sample_weight, alpha=0.9,
                 cmap=plt.cm.bone, edgecolors='black')

    axis.axis('off')
    axis.set_title(title)

# we create 20 points
np.random.seed(0)
# X = np.r_[np.random.randn(10, 2) + [1, 1], np.random.randn(10, 2)]
# y = [1] * 10 + [-1] * 10

X = data.iloc[:, :2] #data.columns != "contraceptive_method"]
y = data.iloc[:,data.columns == "contraceptive_method"].values.ravel()


sample_weight_last_ten = abs(np.random.randn(len(X)))
sample_weight_constant = np.ones(len(X))
# and bigger weights to some outliers
sample_weight_last_ten[15:] *= 5
sample_weight_last_ten[9] *= 15

# for reference, first fit without class weights
# fit the model
clf_weights = svm.SVC()
clf_weights.fit(X, y, sample_weight=sample_weight_last_ten)

clf_no_weights = svm.SVC()
clf_no_weights.fit(X, y)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_decision_function(clf_no_weights, sample_weight_constant, axes[0],
                       "Constant weights")
plot_decision_function(clf_weights, sample_weight_last_ten, axes[1],
                       "Modified weights")

plt.show()