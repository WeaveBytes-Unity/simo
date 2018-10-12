#from  _future_ import division,print_function
import numpy as np
from sklearn import datasets, svm
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read dataset
iris=datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.25, random_state=42)

def evaluate_on_test_data(model=None):
	predictions=model.predict(X_test)
	correct_classification=0
	for i in range(len(y_test)):
		if predictions[i]==y_test[i]:
			correct_classification+=1
	accuracy=100*correct_classification/len(y_test)
	return accuracy


kernels=('linear','poly','rbf')
accuracies=[]
for index,kernel in enumerate(kernels):
	model=svm.SVC(kernel=kernel)
	model.fit(X_train,y_train)
	acc=evaluate_on_test_data(model);
	accuracies.append(acc)
	print("{} % accuracy obtained with kernel= {}".format(acc,kernel))

######################
def calculate_imbalanced_gap(data):
	majority_count = 0
	minority_count = 0
	a = np.array(data)
	print("Data => ", a)
	count_no_use = list(a).count(1)
	count_short_term_use = list(a).count(2)
	count_long_term_use = list(a).count(3)


	# print("count_no_use",count_no_use)
	# print("count_short_term_use",count_short_term_use)
	# print("count_long_term_use",count_long_term_use)

	if(count_no_use > count_short_term_use) and (count_no_use > count_long_term_use):
	    majority_count = count_no_use
	elif(count_short_term_use > count_no_use) and (count_short_term_use > count_long_term_use):
	    majority_count = count_short_term_use
	elif(count_long_term_use > count_no_use) and (count_long_term_use > count_short_term_use):
	    majority_count = count_long_term_use

	if(count_no_use < count_short_term_use) and (count_no_use < count_no_use):
	    minority_count = count_no_use
	elif(count_short_term_use < count_no_use) and (count_short_term_use < count_no_use):
	    minority_count = count_short_term_use
	elif(count_long_term_use < count_no_use) and (count_long_term_use < count_short_term_use):
	    minority_count = count_long_term_use

	I_G = majority_count-minority_count
	print("I_G", I_G)
	return I_G

#first make a model function for modeling with confusion matrix
def model(model,features_train,features_test,labels_train,labels_test):
	alf = model
	alf.fit(features_train, labels_train)
	# clf.fit(features_train,labels_train.values.ravel())

	calculate_imbalanced_gap(labels_train)

	weights = clf.coef
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

## Decision boundary for the SVM
def plot_svc_decision_boundary(svm_clf, features_train, features_test, labels_train, labels_test):
	plt.scatter(features_train[labels_train == 1, 0], features_train[labels_train == 1, 1], color="blue", s=10, label="No-use")
	plt.scatter(features_train[labels_train == 2, 0], features_train[labels_train == 2, 1], color="red", s=10, label="Short-use")
	plt.scatter(features_train[labels_train == 3, 0], features_train[labels_train == 3, 1], color="green", s=10, label="Long-use")

	w = svm_alf.coef_[0]
	a = -w[0] / w[1]
	xxx = np.linspace(-5, 7)
	yyy = a * xx - (svm_alf.intercept_[0]) / w[1]
	plt.plot(xx, yy, "k-", linewidth=2)

	w = svm_alf.coef_[1]
	a = -w[0] / w[1]
	xxx = np.linspace(-5, 7)
	print("xx ====> ", xxx)
	print("xx type ====> ", type(xxx))

	yy = a * xx - (svm_clf.intercept_[1]) / w[1]
	plt.plot(xx, yy, "k--", linewidth=2)

	#plot decision boundary between class 3 and others

	w = svm_clf.coef_[2]
	a = -w[0] / w[1]
	xxx = np.linspace(-5, 7)
	yyy = a * xx - (svm_clf.intercept_[2]) / w[1]

	plt.plot(xx, yy, "k-", linewidth=1)

	plt.show()

# now let us check in the number of Percentage
count_no_use = len(data[iris["class"]==1]) #no use of contraceptive_method is represented by 1
count_long_term_use = len(data[iris["class"]==2]) # long term use of contraceptive_method is represented by 2
count_short_term_use = len(data[iris["class"]==3]) # short term use of contraceptive_method is represented by 3
print("no use of contraceptive_method ",count_no_use)
print("long term use of contraceptive_method",count_long_term_use)
print("short term use of contraceptive_method ",count_short_term_use)
 

######################





#visualise decision boundary
svc=svm.SVC(kernel='linear').fit(X_train,y_train)
rbf_svc=svm.SVC(kernel='rbf',gamma=0.7).fit(X_train,y_train)#gaussian kernel
poly_svc=svm.SVC(kernel='poly',degree=3).fit(X_train,y_train)

# create mesh to plot
h=0.2 # steps in mesh
x_min,x_max=X[:,0].min( ) -1,X[:,0].max()+1
y_min,y_max=X[:,0].min( ) -1,X[:,0].max()+1
xx, yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

#Define title
titles=['SVC with linear kernel','SVC with rbf_kernel','SVC with polynial kernel']

for i, clf in enumerate((svc,rbf_svc,poly_svc)):


    #plot decision boundary
    #plot in the mesh

    plt.figure()

    z=clf.predict(np.c_[xx.ravel(),yy.ravel()])

    #put the reult into a color plot
    z=z.reshape(xx.shape)
    plt.contour(xx,yy,z,cmap=plt.cm.Paired, alpha=0.8)


    #plot also the training points
    plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.ocean)
    plt.xlabel('sepal length')
    plt.ylabel('sepalwidth')
    plt.xlim(xx.min(),xx.max())
    plt.xlim(yy.min(),yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

#plt.show()

    
    
    
    
    
    
   
    