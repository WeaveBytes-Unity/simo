from sklearn.preprocessing import StandardScaler # for preprocessing the data
from sklearn.ensemble import RandomForestClassifier # Random forest classifier
from sklearn.tree import DecisionTreeClassifier # for Decision Tree classifier
from sklearn.svm import SVC # for SVM classification
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split # to split the data
from sklearn.cross_validation import KFold # For cross vbalidation
from sklearn.model_selection import GridSearchCV # for tunnig hyper parameter it will use all combination of given parameters
from sklearn.model_selection import RandomizedSearchCV # same for tunning hyper parameter but will use random combinations of parameters
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
from imblearn.metrics import geometric_mean_score
import pandas as pd # to import csv and for data manipulation
import matplotlib.pyplot as plt # to plot graph
import seaborn as sns # for intractve graphs
import numpy as np # for linear algebra
import datetime # to dela with date and time
import warnings


warnings.filterwarnings('ignore')

colnames =['wife_age','wife_education', 'husband_education', 'children', 'wife_relogion', 
	'wife_working','husband_occupation','standard_of_living', 'media_exposure','contraceptive_method']

data = pd.read_csv('datasets/data.csv', names = colnames, header=None)

# data.info()
# print("Data => ", data.values)

print("=================")
X = np.array(data.ix[:, 1:4].values) #data.columns != "class"]
print("dataaaaa",X)
y = np.array(data.ix[:,data.columns == "contraceptive_method"].values.ravel())
print("dataaa",y)
# features = data.ix[:, :2].values #data.columns != "class"]
# labels   = data.ix[:,data.columns == "class"].values.ravel()

print("features => ", X[0])
print("labels => ", y)
print("features type => ", type(X))
print("labels type => ", type(y))
# print("Value Count =>\n", data.class.value_counts())


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

model = SVC(kernel='linear')
clf = model.fit(X, y)

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVC ')
# Set-up grid for plotting.
X0, X1 = X[:, 0], X[:, 1]
print("Decision _function => ",clf.decision_function(X))
print("Predict => ",clf.predict(X))
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('y label here')
ax.set_xlabel('x label here')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()


####################################################################################3

print("==================")


# preparing data for training and testing as we are going to use different data 
def data_prepration(x):
    #again and again so make a function
    x_features= x.ix[:,x.columns != "contraceptive_method"].values
    x_labels=x.ix[:,x.columns=="contraceptive_method"].values.ravel()
    x_features_train,x_features_test,x_labels_train,x_labels_test = train_test_split(x_features,x_labels,test_size=0.2)
    # print("length of training data", x_features_train.shape)
    # print(len(x_features_train))
    # print("length of test data", x_labels_train.shape)
    # print(len(x_features_test))
    # print("Data group", data.groupby('wife_working').size())
    return(x_features_train,x_features_test,x_labels_train,x_labels_test)

def calculate_imbalanced_gap(data):
	majority_count=0
	minority_count=0
	a= np.array(data)
	print("Data => ", a)
	count_no_use = list(a).count(1)
	count_short_term_use = list(a).count(2)
	count_long_term_use = list(a).count(3)
	print("count_no_use",count_no_use)
	print("count_short_term_use",count_short_term_use)
	print("count_long_term_use",count_long_term_use)

	if(count_no_use > count_short_term_use) and (count_no_use > count_long_term_use):
		majority_count = count_no_use
	elif(count_short_term_use > count_no_use) and (count_short_term_use > count_long_term_use):
		majority_count = count_short_term_use
	elif(count_long_term_use > count_no_use) and (count_long_term_use > count_short_term_use):
		majority_count = count_long_term_use

	if(count_no_use < count_short_term_use) and (count_no_use < count_no_use):
		minority_count = count_no_use
	elif(count_short_term_use > count_no_use) and (count_short_term_use > count_long_term_use):
		minority_count = count_short_term_use
	elif(count_long_term_use < count_no_use) and (count_long_term_use < count_short_term_use):
		minority_count = count_long_term_use

	I_G = majority_count - minority_count
	print("I_G=",I_G)
	return I_G


## first make a model function for modeling with confusion matrix
def model(model,features_train,features_test,labels_train,labels_test):
    clf = model
    clf.fit(features_train, labels_train)
    # clf.fit(features_train,labels_train.values.ravel())

    calculate_imbalanced_gap(labels_train)

    weights = clf.coef_
    bias = clf.intercept_
    # print('Indices of support vectors = ', clf.support_)
    # print('Support vectors = ', clf.support_vectors_[1])
    # print('Number of support vectors for each class = ', clf.n_support_)
    # print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))

    pred = clf.predict(features_test)
    cnf_matrix = confusion_matrix(labels_test, pred)
    # print("cnf_matrix => ", cnf_matrix)
    # print("g-mean", geometric_mean_score(labels_test, pred))
    # print("the accuracy for this method is : ", clf.score(features_test, labels_test))
    # print("the recall score for this model is :", recall_score(labels_test, pred, average=None))
    # print("the recall for this model is :", cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]))
    print("\n----------Classification Report------------------------------------")
    print(classification_report(labels_test, pred))
    print("TP",cnf_matrix[1,1]) 
    print("TN",cnf_matrix[0,0]) 
    print("FP",cnf_matrix[0,1])
    print("FN",cnf_matrix[1,0]) 
    sns.heatmap(cnf_matrix, cmap="coolwarm_r", annot=True, linewidths=0.5)
    # plt.title("Confusion_matrix")
    # plt.xlabel("Predicted_class")
    # plt.ylabel("Real class")
    # plt.show()
  
## Decision boundary for the SVM   
def plot_svc_decision_boundary(svm_clf, features_train, features_test, labels_train, labels_test):
	plt.scatter(features_train[labels_train == 1, 0], features_train[labels_train == 1, 1], color="blue", s=10, label="No-use")
	plt.scatter(features_train[labels_train == 2, 0], features_train[labels_train == 2, 1], color="red", s=10, label="Short-use")
	plt.scatter(features_train[labels_train == 3, 0], features_train[labels_train == 3, 1], color="green", s=10, label="Long-use")

	#plot decision boundary for claass 1 and other
	w = svm_clf.coef_[0]
	b = svm_clf.intercept_[0]
	xx = np.linspace(-5, 7)
	yy = a * xx - (svm_clf.intercept_[0]) / w[1]
	plt.plot(xx, yy, "k-", linewidth=2)

	# plot decision boundary between class 2 and others
	w = svm_clf.coef_[1]
	a = -w[0] / w[1]
	xx = np.linspace(-5, 7)
	print("xx ====> ", xx)
	print("xx type ====> ", type(xx))

	yy = a * xx - (svm_clf.intercept_[1]) / w[1]
	plt.plot(xx, yy, "k--", linewidth=2)

	# plot decision boundary between class 2 and others
	w = svm_clf.coef_[2]
	a = -w[0] / w[1]
	xx = np.linspace(-5, 7)
	yy = a * xx - (svm_clf.intercept_[2]) / w[1]
	plt.plot(xx, yy, "k-", linewidth=1)

	plt.show()

	# now let us check in the number of Percentage
	count_no_use = len(data[data["class"]==1]) #no use of contraceptive_method is represented by 1
	count_long_term_use = len(data[data["class"]==2]) # long term use of contraceptive_method is represented by 2
	count_short_term_use = len(data[data["class"]==3]) # short term use of contraceptive_method is represented by 3
	print("no use of contraceptive_method ",count_no_use)
	print("long term use of contraceptive_method",count_long_term_use)
	print("short term use of contraceptive_method ",count_short_term_use)



	# now we can divided our data into training and test data
	# Call our method data prepration on our dataset
data_train_X,data_test_X,data_train_y,data_test_y=data_prepration(data)
	# columns = data_train_X.columns
	# print("Columns => ", columns)

calculate_imbalanced_gap(data_train_y)
svclassifier = SVC(kernel='linear', C=1.0)

svclassifier.fit(data_train_X, data_train_y)
print('Number of support vectors for each class = ', svclassifier.n_support_)

	# plot_svc_decision_boundary(svclassifier, data_train_X, data_test_X, data_train_y, data_test_y

y_pred = svclassifier.predict(data_test_X)
# print("Prediction => ", y_pred)
score = svclassifier.score(data_test_X, data_test_y)
print("Score => ", score)

os = SMOTE(random_state=0, kind='svm') # We are using SMOTE as the function for oversampling

# now use SMOTE to oversample our train data which have features data_train_X and labels in data_train_y
os_data_X, os_data_y = os.fit_sample(data_train_X, data_train_y)

# clf = RandomForestClassifier(n_estimators=100)
clf = SVC(kernel='linear')
# train data using oversampled data and predict for the test data
model(clf, os_data_X, data_test_X, os_data_y, data_test_y)



	#