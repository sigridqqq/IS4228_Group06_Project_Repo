import numpy as np 
import json 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from scipy import sparse 
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

train_path ='./data/train_processed.csv'
val_path =  './data/val_processed.csv'
test_path = './data/test_processed.csv'

train = pd.read_csv(train_path)
val = pd.read_csv(val_path)
test = pd.read_csv(test_path)
train_fun = train[train.funny_sampled_flag==1]
train_cool = train[train.cool_sampled_flag==1]

for i in [train, train_fun, train_cool, val, test]:
    i.dropna(inplace=True)
    print(i.shape)


train.reset_index(inplace=True)
train_fun.reset_index(inplace=True)
train_cool.reset_index(inplace=True)
val.reset_index(inplace=True)
test.reset_index(inplace=True)


# train_array = np.load('./useful_processed_text/useful_sbert_train.npy')
# val_array = np.load('./useful_processed_text/useful_sbert_val.npy')
# test_array = np.load('./useful_processed_text/useful_sbert_test.npy')

# for i in [train_array, val_array, test_array]:
#     print(i.shape)

# model = 'doc2vec'
model = 'sbert'
with open(f'./text_processing/funny_processed_text/funny_{model}_train.npy', 'rb') as f:
    train_array = np.load(f)
with open(f'./text_processing/funny_processed_text/funny_{model}_val.npy', 'rb') as f:
    val_array = np.load(f)
with open(f'./text_processing/funny_processed_text/funny_{model}_test.npy', 'rb') as f:
    test_array = np.load(f)

for i in [train_array, val_array, test_array]:
    print(i.shape)



# from sklearn import ensemble
from sklearn import tree, ensemble, model_selection
from sklearn.metrics import mean_squared_error
dt_estimator = tree.DecisionTreeClassifier(splitter='random',max_depth=5,min_samples_split=50,min_samples_leaf=5)
dt_estimator.fit(train_array, train.funny_label)
y_train_funny = dt_estimator.predict(train_array)
y_val_funny = dt_estimator.predict(val_array)
y_test_funny = dt_estimator.predict(test_array)
print("accuracy score: ", accuracy_score(train.funny_label, y_train_funny))
print("confusion matrix: ")
print(confusion_matrix(train.funny_label, y_train_funny))
print("accuracy score: ", accuracy_score(val.funny_label, y_val_funny))
print("confusion matrix: ")
print(confusion_matrix(val.funny_label, y_val_funny))
print("accuracy score: ", accuracy_score(test.funny_label, y_test_funny))
print("confusion matrix: ")
print(confusion_matrix(test.funny_label, y_test_funny))


dt_estimator = ensemble.GradientBoostingClassifier(learning_rate=0.1)
dt_estimator.fit(train_array, train.funny_label)
y_train_funny = dt_estimator.predict(train_array)
y_val_funny = dt_estimator.predict(val_array)
y_test_funny = dt_estimator.predict(test_array)
print("accuracy score: ", accuracy_score(train.funny_label, y_train_funny))
print("confusion matrix: ")
print(confusion_matrix(train.funny_label, y_train_funny))
print("accuracy score: ", accuracy_score(val.funny_label, y_val_funny))
print("confusion matrix: ")
print(confusion_matrix(val.funny_label, y_val_funny))
print("accuracy score: ", accuracy_score(test.funny_label, y_test_funny))
print("confusion matrix: ")
print(confusion_matrix(test.funny_label, y_test_funny))


dt_estimator = ensemble.AdaBoostClassifier(learning_rate=0.1)
dt_estimator.fit(train_array, train.funny_label)
y_train_funny = dt_estimator.predict(train_array)
y_val_funny = dt_estimator.predict(val_array)
y_test_funny = dt_estimator.predict(test_array)
print("accuracy score: ", accuracy_score(train.funny_label, y_train_funny))
print("confusion matrix: ")
print(confusion_matrix(train.funny_label, y_train_funny))
print("accuracy score: ", accuracy_score(val.funny_label, y_val_funny))
print("confusion matrix: ")
print(confusion_matrix(val.funny_label, y_val_funny))
print("accuracy score: ", accuracy_score(test.funny_label, y_test_funny))
print("confusion matrix: ")
print(confusion_matrix(test.funny_label, y_test_funny))



from xgboost import XGBClassifier

dt_estimator = XGBClassifier()
dt_estimator.fit(train_array, train.funny_label)
y_train_funny = dt_estimator.predict(train_array)
y_val_funny = dt_estimator.predict(val_array)
y_test_funny = dt_estimator.predict(test_array)
print("accuracy score: ", accuracy_score(train.funny_label, y_train_funny))
print("confusion matrix: ")
print(confusion_matrix(train.funny_label, y_train_funny))
print("accuracy score: ", accuracy_score(val.funny_label, y_val_funny))
print("confusion matrix: ")
print(confusion_matrix(val.funny_label, y_val_funny))
print("accuracy score: ", accuracy_score(test.funny_label, y_test_funny))
print("confusion matrix: ")
print(confusion_matrix(test.funny_label, y_test_funny))
