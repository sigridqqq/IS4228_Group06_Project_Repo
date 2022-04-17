import numpy as np 
import json 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from scipy import sparse 
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

# train_path = '/content/drive/My Drive/yelp_review_data/preprocessing/yelp2017_train_processed.csv'
# val_path = '/content/drive/My Drive/yelp_review_data/preprocessing/yelp2017_val_processed.csv'
# test_path = '/content/drive/My Drive/yelp_review_data/preprocessing/yelp2017_test_processed.csv'

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
# for i in [train, val, test]:
#     i.dropna(inplace=True)
#     print(i.shape)

train.reset_index(inplace=True)
train_fun.reset_index(inplace=True)
train_cool.reset_index(inplace=True)
val.reset_index(inplace=True)
test.reset_index(inplace=True)


# model = 'count_2gram'
# model = 'hashing_2gram'
model = 'tfidf_2gram'
with open(f'./useful_processed_text/useful_{model}_train.npz', 'rb') as f:
    train_array = sparse.load_npz(f)
with open(f'./useful_processed_text/useful_{model}_eval.npz', 'rb') as f:
    val_array = sparse.load_npz(f)
with open(f'./useful_processed_text/useful_{model}_test.npz', 'rb') as f:
    test_array = sparse.load_npz(f)
# train_array = np.load('./useful_processed_text/useful_sbert_train.npy')
# val_array = np.load('./useful_processed_text/useful_sbert_val.npy')
# test_array = np.load('./useful_processed_text/useful_sbert_test.npy')
# train_array = np.load('./cool_processed_text/cool_sbert_train.npy')
# val_array = np.load('./cool_processed_text/cool_sbert_val.npy')
# test_array = np.load('./cool_processed_text/cool_sbert_test.npy')
# model = 'sbert'
# model = 'hashing_1gram'
# with open(f'./cool_processed_text/cool_{model}_train.npz', 'rb') as f:
#     train_array = np.load(f)#,allow_pickle=True,fix_imports=True,encoding='latin1'
# with open(f'./cool_processed_text/cool_{model}_val.npz', 'rb') as f:
#     val_array = np.load(f)
# with open(f'./cool_processed_text/cool_{model}_test.npz', 'rb') as f:
#     test_array = np.load(f)

for i in [train_array, val_array, test_array]:
    print(i.shape)

# model = 'sbert'
# with open(f'./text_processing/useful_processed_text/useful_{model}_train.npy', 'rb') as f:
#     train_array = np.load(f)
# with open(f'./text_processing/useful_processed_text/useful_{model}_val.npy', 'rb') as f:
#     val_array = np.load(f)
# with open(f'./text_processing/useful_processed_text/useful_{model}_test.npy', 'rb') as f:
#     test_array = np.load(f)

# for i in [train_array, val_array, test_array]:
#     print(i.shape)



from sklearn import ensemble
dt_estimator = ensemble.GradientBoostingClassifier(learning_rate=0.1)
dt_estimator.fit(train_array, train.useful_label)
y_train_useful = dt_estimator.predict(train_array)
y_val_useful = dt_estimator.predict(val_array)
y_test_useful = dt_estimator.predict(test_array)
print("accuracy score: ", accuracy_score(train.useful_label, y_train_useful))
print("confusion matrix: ")
print(confusion_matrix(train.useful_label, y_train_useful))
print("accuracy score: ", accuracy_score(val.useful_label, y_val_useful))
print("confusion matrix: ")
print(confusion_matrix(val.useful_label, y_val_useful))
print("accuracy score: ", accuracy_score(test.useful_label, y_test_useful))
print("confusion matrix: ")
print(confusion_matrix(test.useful_label, y_test_useful))




dt_estimator = ensemble.AdaBoostClassifier(learning_rate=0.1)
dt_estimator.fit(train_array, train.useful_label)
y_train_useful = dt_estimator.predict(train_array)
y_val_useful = dt_estimator.predict(val_array)
y_test_useful = dt_estimator.predict(test_array)
print("accuracy score: ", accuracy_score(train.useful_label, y_train_useful))
print("confusion matrix: ")
print(confusion_matrix(train.useful_label, y_train_useful))
print("accuracy score: ", accuracy_score(val.useful_label, y_val_useful))
print("confusion matrix: ")
print(confusion_matrix(val.useful_label, y_val_useful))
print("accuracy score: ", accuracy_score(test.useful_label, y_test_useful))
print("confusion matrix: ")
print(confusion_matrix(test.useful_label, y_test_useful))



from xgboost import XGBClassifier

dt_estimator = XGBClassifier()
dt_estimator.fit(train_array, train.useful_label)
y_train_useful = dt_estimator.predict(train_array)
y_val_useful = dt_estimator.predict(val_array)
y_test_useful = dt_estimator.predict(test_array)
print("accuracy score: ", accuracy_score(train.useful_label, y_train_useful))
print("confusion matrix: ")
print(confusion_matrix(train.useful_label, y_train_useful))
print("accuracy score: ", accuracy_score(val.useful_label, y_val_useful))
print("confusion matrix: ")
print(confusion_matrix(val.useful_label, y_val_useful))
print("accuracy score: ", accuracy_score(test.useful_label, y_test_useful))
print("confusion matrix: ")
print(confusion_matrix(test.useful_label, y_test_useful))




