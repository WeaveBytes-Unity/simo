from collections import Counter
from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline  import make_pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.svm import SVC  
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def print_results(headline, true_value, pred):
	print(headline)
	print("accuracy:{}".format(accuracy_score(true_value, pred)))
	print("precision:{}".format(precision_score(true_value, pred)))
	print("recall:{}".format(recall_score(true_value, pred)))
	print("f1:{}".format(f1_score(true_value, pred)))


# our classifier to use
classifier = RandomForestClassifier

data = fetch_datasets()['wine_quality']
print(data)

# splitting data into training and tets sets 
X_train, X_test, y_train, y_test  = train_test_split(data['data'], data['target'], random_state =2)

# build normal model
pipeline = make_pipeline(classifier(random_state=42))
model = pipeline.fit(X_train, y_train)
prediction = model.predict(X_test)

# build model with SMOTE imblearn
smote_pipeline = make_pipeline_imb(SMOTE(random_state=4),classifier(random_state=42))
smote_model = pipeline.fit(X_train, y_train)
smote_prediction = model.predict(X_test)

# build model woth undersampling
nearmiss_pipeline = make_pipeline_imb(NearMiss(random_state= 42), classifier(random_state=42))
nearmiss_model = nearmiss_pipeline.fit(X_train, y_train)
nearmiss_prediction = nearmiss_model.predict(X_test)


# print information about both models
print()
print("normal data distribuition:{}".format(Counter(data['target'])))
X_smote,y_smote = SMOTE().fit_sample(data['data'], data['target'])
print("SMOTE data distribuition:{}".format(Counter(y_smote)))
X_nearmiss, y_nearmiss = NearMiss().fit_sample(data['data'],data['target'])
print("NearMiss data distribuition:{}".format(Counter(y_nearmiss)))

# classification reports
print(classification_report(y_test, prediction))
print(classification_report_imbalanced(y_test, smote_prediction))

print()
print("normal pipeline score {}".format(pipeline.score(X_test, y_test)))
# print("SMOTE pipeline score {}".format(smote_pipeline.score(X_test, y_test)))
print("NearMiss  pipeline score {}".format(nearmiss_pipeline.score(X_test, y_test)))

print()
print_results('normal classifictaion', y_test, prediction)
print()
print_results("SMOTE classification", y_test, smote_prediction)
print()
print_results("Nearmiss classification", y_test, nearmiss_prediction)