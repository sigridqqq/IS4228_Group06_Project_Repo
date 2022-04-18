# BT4222_Group06_Project_Repo


In this project, we closely studied the current storage of reviews on Yelp and build models to predict lables for reviews - they can be labeled as funny, cool and/or useful. We performed data analysis on the full review data first. Then we processed the data for model training. We believe by providing real-time review feedback, Yelp users would be motivated to contribute higher quality reviews on the platform.


Description of code files in this repo:

| Folder Name             |  File Name            | Description         |
| ----------------------- |---------------------- |---------------------|
| Data Preprocessing      |undersample.ipynb      | JupyterNotebook used to create binary labels for the columns <br />to be predicted undersample training dataset for funny and cool prediction |
| Embedding               |doc2vec.py             | Python file used to generate sentence embedding using Doc2Vec |
| Embedding               |sbert.py               | Python file used to generate sentence embedding using SentenceBERT |
| Embedding               |vectorizer.py          | Python file used to generate text representation using Tfidf, Hashing, and Count Vecotrizer |
