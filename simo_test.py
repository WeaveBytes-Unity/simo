import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt 

filename = "datasets/data.csv" 

contraFrame = pd.read_csv(filename,header=None)
column_names = ["Wife's age","Wife's education","Husband's education","Number of children born" \
,"Wife's religion","Wife's now working","Husband's occupation","Standard of living" \
,"Media exposure","Contraceptive Method Used"]

contraFrame.columns = column_names
contraFrame = contraFrame.iloc[np.random.permutation(len(contraFrame))]
target_label = contraFrame["Contraceptive Method Used"]
print(target_label)
del contraFrame["Contraceptive Method Used"] 

train_data,test_data,train_labels,test_labels = cross_validation.train_test_split(contraFrame.values,target_label.values,test_size=0.2)
print('length of traing data', len(train_data))
print('length of test data ', len(test_data))




