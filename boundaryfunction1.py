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

plt.show()

    
    
    
    
    
    
   
    