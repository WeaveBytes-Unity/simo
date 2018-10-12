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

# preparing data for training and testing as we are going to use different data 
def data_prepration(x):
    #again and again so make a function
    x_features= x.ix[:,x.columns != "contraceptive_method"]
    x_labels=x.ix[:,x.columns=="contraceptive_method"]
    x_features_train,x_features_test,x_labels_train,x_labels_test = train_test_split(x_features,x_labels,test_size=0.2)
    print("length of training data", x_features_train.shape)
    print(len(x_features_train))
    print("length of test data", x_labels_train.shape)
    print(len(x_features_test))
    print("Data group", data.groupby('wife_working').size())
    return(x_features_train,x_features_test,x_labels_train,x_labels_test)

def calculate_imbalanced_gap(data):
    a= np.array(data)
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
    elif(count_short_term_use < count_no_use) and (count_short_term_use < count_no_use):
        minority_count = count_short_term_use
    elif(count_long_term_use < count_no_use) and (count_long_term_use < count_short_term_use):
        minority_count = count_long_term_use

    I_G = majority_count-minority_count
    print("I_G", I_G)
    return I_G

## first make a model function for modeling with confusion matrix
def model(model,features_train,features_test,labels_train,labels_test):
    clf = model
    clf.fit(features_train,labels_train.values.ravel())
    pred = clf.predict(features_test)
    cnf_matrix = confusion_matrix(labels_test,pred)
    print("g-mean", geometric_mean_score(labels_test, pred))
    print("the accuracy for this method is : ",clf.score(features_test, labels_test))
    print("the recall for this model is :",cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]))
    print("\n----------Classification Report------------------------------------")
    print(classification_report(labels_test,pred))
    fpr, tpr, thresholds = metrics.roc_curve(labels_test, pred, pos_label=2)
    print("auc",metrics.auc(fpr, tpr))
    fig = plt.figure(figsize=(6,3))# to plot the graph
    print("TP",cnf_matrix[1,1]) 
    print("TN",cnf_matrix[0,0]) 
    print("FP",cnf_matrix[0,1])
    print("FN",cnf_matrix[1,0]) 
    sns.heatmap(cnf_matrix,cmap="coolwarm_r",annot=True,linewidths=0.5)
    # plt.title("Confusion_matrix")
    # plt.xlabel("Predicted_class")
    # plt.ylabel("Real class")
    # plt.show()
    

# now let us check in the number of Percentage
count_no_use = len(data[data["contraceptive_method"]==1]) #no use of contraceptive_method is represented by 1
count_long_term_use = len(data[data["contraceptive_method"]==2]) # long term use of contraceptive_method is represented by 2
count_short_term_use = len(data[data["contraceptive_method"]==3]) # short term use of contraceptive_method is represented by 3
# print("no use of contraceptive_method ",count_no_use)
# print("long term use of contraceptive_method",count_long_term_use)
# print("short term use of contraceptive_method ",count_short_term_use)


# now we can divided our data into training and test data
# Call our method data prepration on our dataset
data_train_X,data_test_X,data_train_y,data_test_y=data_prepration(data)
columns = data_train_X.columns
print("Columns => ", columns)
calculate_imbalanced_gap(data_train_y)
svclassifier = SVC(kernel='linear')  

svclassifier.fit(data_train_X, data_train_y)
y_pred = svclassifier.predict(data_test_X)
# print("Prediction => ", y_pred)
score = svclassifier.score(data_test_X, data_test_y)
print("Score => ", score)
# print("confusion_matrix => ", confusion_matrix(data_test_y,y_pred))
# print("classification_report => ", classification_report(data_test_y,y_pred)) 


os = SMOTE(random_state=0, kind='svm') # We are using SMOTE as the function for oversampling

# now use SMOTE to oversample our train data which have features data_train_X and labels in data_train_y
os_data_X, os_data_y = os.fit_sample(data_train_X, data_train_y)
os_data_X = pd.DataFrame(data=os_data_X, columns=columns )
os_data_y= pd.DataFrame(data=os_data_y, columns=["contraceptive_method"])
# we can Check the numbers of our data
# print("length of oversampled data is ",len(os_data_X))
# print("no use of contraceptive_method  in oversampled data",len(os_data_y[os_data_y["contraceptive_method"]==1]))
# print("long term use of contraceptive_method in oversampled data",len(os_data_y[os_data_y["contraceptive_method"]==2]))
# print("short term use of contraceptive_method in oversampled data",len(os_data_y[os_data_y["contraceptive_method"]==3]))
# print("Proportion of no use of contraceptive_method data  in oversampled data is ",len(os_data_y[os_data_y["contraceptive_method"]==1])/len(os_data_X))
# print("Proportion of long term use of contraceptive_method in oversampled data is ",len(os_data_y[os_data_y["contraceptive_method"]==2])/len(os_data_X))
# print("Proportion of short term use of contraceptive_method in oversampled data is ",len(os_data_y[os_data_y["contraceptive_method"]==3])/len(os_data_X))



# Now start modeling
clf= RandomForestClassifier(n_estimators=100)
# train data using oversampled data and predict for the test data
model(clf,os_data_X,data_test_X,os_data_y,data_test_y)