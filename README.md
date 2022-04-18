# BT4222_Group06_Project_Repo


In this project, we closely studied the current storage of reviews on Yelp and build models to predict labels for reviews - they can be labeled as funny, cool and/or useful. We performed data analysis on the full review data first. Then we processed the data for model training. We believe by providing real-time review feedback, Yelp users would be motivated to contribute higher quality reviews on the platform.

Please refer to the documentation.pdf for a detailed explanation on how the files are used.

Description of code files in this repo:

| Folder Name             |  File Name             | Description         |
| ----------------------- |----------------------- |---------------------|
| Data Analysis           |EDA.ipynb               | JupyterNotebook for EDA|
| Data Preprocessing      |Data Preprocessing.ipynb| JupyterNotebook for data preprocessing|
| Data Preprocessing      |detect_language.py      | Python file for review language detection|
| Data Preprocessing      |undersample.ipynb       | JupyterNotebook for data preprocessing|
| Embedding               |doc2vec.py              | Python file for sentence embedding generation using Doc2Vec|
| Embedding               |sbert.py                | Python file for sentence embedding generation using SentenceBERT|
| Embedding               |vectorizer.py           | Python file for text representation generation using Tfidf, Hashing, and Count Vecotrizer|
| Model Running           |ALBERT_fine_tune.py     | Python file for classification with fine-tuned ALBERT|
| Model Running           |BERT_fine_tune.py       | Python file for classification with fine-tuned BERT|
| Model Running           |F1_calculation.py       | Python file for computing F1 performance with input data|
| Model Running           |Model_Run_{Cool/Funny/Useful}_npy.py       | Python file for classification with sbert or doc2vec embedding|
| Model Running           |Model_Run_{Cool/Funny/Useful}_npz.py       | Python file for classification with text representation by vectorizers|


Link to download full review CSV: https://drive.google.com/file/d/17uPXOVdgIPFMGH8uScdK6qclE0sby7Ks/view?usp=sharing

Official Yelp Reivew JSON file download: https://www.yelp.com/dataset 
