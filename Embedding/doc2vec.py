from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd 
import numpy as np 
from tqdm import tqdm

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

print("------- current model: Doc2Vec -------")
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

print("currently tokenizing train text")
tokenized_sent = []
for s in tqdm(train['cleaned_text']):
    tokenized_sent.append(word_tokenize(s.lower()))

print("currently encoding val text")
tokenized_sent_val = []
for s in tqdm(val['cleaned_text']):
    tokenized_sent_val.append(word_tokenize(s.lower()))

print("currently encoding test text")
tokenized_sent_test = []
for s in tqdm(test['cleaned_text']):
    tokenized_sent_test.append(word_tokenize(s.lower()))

model = 'doc2vec'
vector_size = 60
doc2vec_filename = f'saved_model/{category}_doc2vec_model.sav'

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
tagged_text = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
print("currently learning doc2vec model")
doc2vec = Doc2Vec(tagged_text, vector_size = vector_size, window = 2, min_count = 1, epochs = 100)
pickle.dump(doc2vec, open(doc2vec_filename, 'wb'))

print("currently loading doc2vec model")
doc2vec = pickle.load(open(doc2vec_filename, 'rb'))

print("currently encoding train text")
train_vector_array = np.array([])
resultant_train_vector_array = np.array([])

total_len = len(tokenized_sent)
for i in tqdm(range(total_len)): 
    if i%20000 == 0:
        resultant_train_vector_array = np.append(resultant_train_vector_array, train_vector_array)
        train_vector_array = np.array([])
        train_vector_array = np.append(train_vector_array, doc2vec.infer_vector(tokenized_sent[i]))
    else:
        train_vector_array = np.append(train_vector_array, doc2vec.infer_vector(tokenized_sent[i]))
resultant_train_vector_array = np.append(resultant_train_vector_array, train_vector_array)
train_vector_array = resultant_train_vector_array.reshape([-1, vector_size])

print(train_vector_array.shape)
with open(f'processed_text/{category}_{model}_train.npy', 'wb') as f:
    np.save(f, train_vector_array)


print("currently encoding val text")
val_vector_array = np.array([])
for i in tqdm(tokenized_sent_val):
    val_vector_array = np.append(val_vector_array, doc2vec.infer_vector(i))
val_vector_array = val_vector_array.reshape([-1, vector_size])
with open(f'processed_text/{category}_{model}_val.npy', 'wb') as f:
    np.save(f, val_vector_array)

print("currently encoding test text")
test_vector_array = np.array([])
for i in tqdm(tokenized_sent_test):
    test_vector_array = np.append(test_vector_array, doc2vec.infer_vector(i))
test_vector_array = test_vector_array.reshape([-1, vector_size])
with open(f'processed_text/{category}_{model}_test.npy', 'wb') as f:
    np.save(f, test_vector_array)

# to load the saved embedding 
# with open(f'processed_text/{model}_train.npy', 'rb') as f:
    # train_vector_array_reload = np.load(f)

print('doc2vec embedding shape for train, val, test is respectively: ', train_vector_array.shape, val_vector_array.shape, test_vector_array.shape)
