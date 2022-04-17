# Detect language

import pandas as pd 
from langdetect import detect
import pickle

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

train = pd.read_csv("yelp2017_train.csv")
val   = pd.read_csv("yelp2017_val.csv")
test  = pd.read_csv("yelp2017_test.csv")

df = pd.concat([train, val,test], axis=0)
df.dropna(inplace=True)
df['language'] = ''

print('start detect')

for i in range(len(df)):
    print(i)
    df.iloc[i,-1] = detect(df.iloc[i,-2])

save_obj(df, 'df_lang')

# Detect Chinese, Japanese and Korean

import re
df2 = df2.reset_index()
indexlist = []
for x in df2.index:
    if re.search(u'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]', df2.iloc[x,-1]):
        indexlist.append(x)

update_df = df2.drop(indexlist)

#####remove chinese, japanese, korean#########

df1 = update_df
df1 = df1.drop(df1[(df1.language == 'zh-cn') | (df1.language == 'zh-tw') | (df1.language == 'ko') | (df1.language == 'ja') ].index)

import nltk
nltk.download('words')
words = set(nltk.corpus.words.words())

df1.drop(columns=df1.columns[0], axis=1, inplace=True)  #drop the first col if it is the repeated indexing 
df1 = df1.reset_index(drop=True) #reset index to make sure the index match the row number

#to remove non english words in the rest lines
removed_text = []
for i in range(len(df1)):
  #print(i)
  #if not i == True:
    #continue
  cleaned_text = df1['cleaned_text'][i]
  removed_text.append(" ".join(w for w in nltk.wordpunct_tokenize(cleaned_text) \

                     if w.lower() in words or not w.isalpha()))

df1['removed_text']=removed_text
df1.to_csv('removed.csv',index=False)

