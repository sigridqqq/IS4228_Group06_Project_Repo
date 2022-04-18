import pandas as pd 
from langdetect import detect
import pickle

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

train = pd.read_csv("/home/q/qshichao/BT4222/Archive/yelp2017_train.csv")
val   = pd.read_csv("/home/q/qshichao/BT4222/Archive/yelp2017_val.csv")
test  = pd.read_csv("/home/q/qshichao/BT4222/Archive/yelp2017_test.csv")

train = train.drop('Unnamed: 0',1)
val   = val.drop('Unnamed: 0',1)
test  = test.drop('Unnamed: 0',1)

df = pd.concat([train, val,test], axis=0)
df.dropna(inplace=True)
df['language'] = ''

print('start detect')

print('total number of jocs = ' + str(len(df)))
print()

for i in range(len(df)):
    print(i)
    try:
        language = detect(df.iloc[i,-8])
        df.iloc[i,-1] = language
    except:
        df.iloc[i,-1] = "error"
        print("This row throws and error: ", df.iloc[i,-8])


save_obj(df, 'original_review_language_detect')