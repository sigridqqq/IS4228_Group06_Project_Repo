from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd 
import numpy as np 
from tqdm import tqdm

# change category value accordingly
category = 'cool'
print(f"currently running for {category}")

print("load train, val, test data")
if category == 'useful':
    train_path = 'data/train_processed.csv'
elif category == 'funny':
    train_path = 'data/funny_train_processed.csv'
else:
    train_path = 'data/cool_train_processed.csv'

val_path = 'data/val_processed.csv'
test_path = 'data/test_processed.csv'
train = pd.read_csv(train_path)
val = pd.read_csv(val_path)
test = pd.read_csv(test_path)

print("check data shape, drop rows with na values")
print("before dropna")
for i in [train, val, test]:
    print(i.shape)
for i in [train, val, test]:
    i.dropna(inplace=True)
print("after dropna")
for i in [train, val, test]:
    print(i.shape)


print("------- current model: SentenceBERT -------")
model = 'sbert'
# SentenceBERT_filename = f'saved_model/{category}_sbert_model.sav'
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
# pickle.dump(sbert_model, open(SentenceBERT_filename, 'wb'))

# for faster loading of sbert_model, can use the following code from the second time of running this script onwards
# sbert_model = pickle.load(open(sbert_filename, 'rb'))

print("currently encoding train text")
train_sentence_embeddings = sbert_model.encode(list(train['cleaned_text']))
with open(f'text_processing/{category}_processed_text/{category}_{model}_train.npy', 'wb') as f:
    np.save(f, train_sentence_embeddings)

print("currently encoding val text")
val_sentence_embeddings = sbert_model.encode(list(val['cleaned_text']))
with open(f'text_processing/{category}_processed_text/{category}_{model}_val.npy', 'wb') as f:
    np.save(f, val_sentence_embeddings)

print("currently encoding test text")
test_sentence_embeddings = sbert_model.encode(list(test['cleaned_text']))
with open(f'text_processing/{category}_processed_text/{category}_{model}_test.npy', 'wb') as f:
    np.save(f, test_sentence_embeddings)

print('BERT embedding length', len(train_sentence_embeddings[0]))
print('BERT embedding shape for train, val, test is respectively: ', train_sentence_embeddings.shape, val_sentence_embeddings.shape, test_sentence_embeddings.shape)
