# import numpy as np
# import pandas as pd # to import csv and for data manipulation
# from sklearn.cross_validation import train_test_split # to split the data
# from imblearn.over_sampling import SMOTE
# from sklearn.svm import SVC # for SVM classification

# import matplotlib.pyplot as plt
# from sklearn import svm
# from sklearn.datasets import make_blobs


# colnames =['wife_age','wife_education', 'husband_education', 'children', 'wife_relogion', 
# 	'wife_working','husband_occupation','standard_of_living', 'media_exposure','contraceptive_method']

# data = pd.read_csv('datasets/data.csv', names = colnames, header=None)
# features= data.ix[:,data.columns != "contraceptive_method"]
# labels=data.ix[:,data.columns=="contraceptive_method"]

# print("Value Count =>\n", data.contraceptive_method.value_counts())

# x_train1,x_test1,y_train1,y_test1 = train_test_split(features, labels,test_size=0.2, random_state=12)
# x_train,x_test,y_train,y_test = train_test_split(x_train1, y_train1,test_size=0.2, random_state=12)

# print("=================")
# sm = SMOTE(random_state=12, kind='svm', ratio = 1.0)
# data_X, data_y = sm.fit_sample(x_train, y_train)

# # clf_rf = RandomForestClassifier(n_estimators=25, random_state=12)
# clf = SVC(kernel='linear')
# clf.fit(data_X, data_y)
# data = clf.support_vectors_

# plt.scatter(data[:, 0], data[:, 1], c=None, s=30, cmap=None) #Error
# # plot the decision function
# ax = plt.gca()
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# # create grid to evaluate model
# xx = np.linspace(xlim[0], xlim[1], 30)
# yy = np.linspace(ylim[0], ylim[1], 30)
# YY, XX = np.meshgrid(yy, xx)
# xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Z = clf.decision_function(xy).reshape(XX.shape)
# # plot decision boundary and margins
# ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
#            linestyles=['--', '-', '--'])
# # plot support vectors
# ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
#            linewidth=1, facecolors='none')
# plt.show()

#========================
# we create 40 separable points
# X, y = make_blobs(n_samples=40, centers=2, random_state=6)

# # fit the model, don't regularize for illustration purposes
# clf = svm.SVC(kernel='linear', C=1000)
# clf.fit(X, y)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
# # plot the decision function
# ax = plt.gca()
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# # create grid to evaluate model
# xx = np.linspace(xlim[0], xlim[1], 30)
# yy = np.linspace(ylim[0], ylim[1], 30)
# YY, XX = np.meshgrid(yy, xx)
# xy = np.vstack([XX.ravel(), YY.ravel()]).T
# Z = clf.decision_function(xy).reshape(XX.shape)
# # plot decision boundary and margins
# ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
#            linestyles=['--', '-', '--'])
# # plot support vectors
# ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
#            linewidth=1, facecolors='none')
# plt.show()


#=====================
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd # to import csv and for data manipulation
from sklearn.cross_validation import train_test_split 
from imblearn.over_sampling import SMOTE

# we create 40 separable points
# np.random.seed(0)
# X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
# Y = [0] * 20 + [1] * 20
# # print(X.shape)

filename = "datasets/data.csv" 

contraFrame = pd.read_csv(filename,header=None)
colnames =['wife_age','wife_education', 'husband_education', 'children', 'wife_relogion', 
	'wife_working','husband_occupation','standard_of_living', 'media_exposure','contraceptive_method']
data = pd.read_csv(filename, names = colnames, header=None)

print(type(data))
X = data.iloc[:, data.columns != "contraceptive_method"].values # :2].values
Y = data.iloc[:,data.columns == "contraceptive_method"].values.ravel()

print("X",X)
print("X type ",type(X))
print("Y",Y)
print("Y type ",type(Y))

# x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.2, random_state=12)
# print("x_train",x_train.shape)
# print("y_train",y_train.shape)

def plot_data(inputs,targets,weights):
    # fig config
    plt.figure(figsize=(10,6))
    plt.grid(True)

    #plot input samples(2D data points) and i have two classes. 
    #one is +1 and second one is -1, so it red color for +1 and blue color for -1
    for input,target in zip(inputs,targets):
        plt.plot(input[0],input[1],'ro' if (target == 1.0) else 'bo')

    # Here i am calculating slope and intercept with given three weights
    for i in np.linspace(np.amin(inputs[:,:1]),np.amax(inputs[:,:1])):
        slope = -(weights[0]/weights[2])/(weights[0]/weights[1])  
        intercept = -weights[0]/weights[2]

        #y =mx+c, m is slope and c is intercept
        y = (slope*i) + intercept
        plt.plot(i, y,'ko')

    plt.show()

# # figure number
# fignum = 1
# fig, axis = plt.subplots()

# fit the model
# for name, penalty in (('unreg', 1), ('reg', 0.05)):
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
clf1 = svm.SVC(kernel='linear')
clf1.fit(x_train, y_train)

os = SMOTE(random_state=0, kind='svm')
os_data_X, os_data_y = os.fit_sample(x_train, y_train)
clf = svm.SVC(kernel='linear')
clf.fit(os_data_X, os_data_y)
weights = clf.coef_

plot_data(os_data_X, os_data_y, weights)

# xmin = 15
# xmax = 50

# # get the separating hyperplane
# w = clf.coef_[0]
# a = -w[0] / w[1]
# xx = np.linspace(xmin, xmax, 200)
# yy = a * xx - (clf.intercept_[0]) / w[1]

# print("xx =>",xx)
# print("yy =>",yy)

# # plot the parallels to the separating hyperplane that pass through the
# # support vectors (margin away from hyperplane in direction
# # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
# # 2-d.

# margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
# yy_down = yy - np.sqrt(1 + a ** 2) * margin
# yy_up = yy + np.sqrt(1 + a ** 2) * margin

# # plot the line, the points, and the nearest vectors to the plane
# # plt.figure(fignum, figsize=(8, 8))
# # plt.clf()
# # plt.plot(xx, yy, 'k-') # 
# # plt.plot(xx, yy_down, 'k--')
# # plt.plot(xx, yy_up, 'k--')

# # plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],s=5,facecolors='black')
# # plt.scatter(X[:, 0],X[:, 1] ,c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

# plt.scatter(os_data_X[os_data_y == 1, 0], os_data_X[os_data_y == 1, 1], color="blue", s=10, label="No-use")
# plt.scatter(os_data_X[os_data_y == 2, 0], os_data_X[os_data_y == 2, 1], color="red", s=10, label="Short-use")
# plt.scatter(os_data_X[os_data_y == 3, 0], os_data_X[os_data_y == 3, 1], color="green", s=10, label="Long-use")

# plt.plot(xx, yy, 'k-', color="black", label="Model")

# # plt.axis('tight')
# # x_min = 15
# # x_max = 50
# # y_min = 0
# # y_max = 5
# # x_min = -14.8
# # x_max = 14.2
# # y_min = -16
# # y_max = 16

# # XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
# # Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])

# # # Put the result into a color plot
# # Z = Z.reshape(XX.shape)
# # plt.figure(fignum, figsize=(8, 8))
# # plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

# # plt.xlim(x_min, x_max)
# # plt.ylim(y_min, y_max)

# # plt.xticks(())
# # plt.yticks(())
# # fignum = fignum + 1
# axis.legend()
# plt.show()