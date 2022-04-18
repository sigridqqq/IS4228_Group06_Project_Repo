import pandas as pd 
from langdetect import detect
import pickle

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# To download yelp_review.csv, go to https://drive.google.com/file/d/17uPXOVdgIPFMGH8uScdK6qclE0sby7Ks/view?usp=sharing
# Change filepath here accordingly

df = pd.read_csv("/home/q/qshichao/BT4222/Archive/yelp_review.csv")
df.dropna(inplace=True)
df['language'] = ''

print('start detect')

print('total number of jobs = ' + str(len(df)))
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
# To download 'original_review_language_detect', go to https://drive.google.com/file/d/1e9pqur4iZUcRmi9hMK49U21m61kLCQ6S/view?usp=sharing