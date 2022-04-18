'''
In this python script we are using huggingface's pretrained ALBERT model for fine tuning.

The entire scripts take about 45 hours to complete.

For code error check, reduce the number of data rows passed into trainer.

Also make sure the packages are properly installed before running the script

'''


import pandas as pd
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, DefaultDataCollator, TFAutoModelForSequenceClassification
import tensorflow as tf

# To download train.csv, go to https://drive.google.com/file/d/1dkY35gnkHGUDRi4flC8rV3vk_P1LceOL/view?usp=sharing
# To download val.csv,   go to https://drive.google.com/file/d/1SCzPAb_CAw7AOeLSmPYRsSl-hEmzpKU6/view?usp=sharing
# To download test.csv,  go to https://drive.google.com/file/d/1xMcG0yjZKEiU85pt92IHIDDWWPZriWcR/view?usp=sharing 

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

def label_race(row,column):
   if row[column] == 0 :
      return 0
   if row[column] > 0 :
      return 1


# Here we create a new column named 'useful_bin' in the dataset 
# Those reviews with 0 useful votes are regarded as unuseful, and will have a value 0 in column 'useful_bin'
# Those reviews with more than 0 useful votes are regarded as useful, and will have a value 1 in column 'useful_bin'

train['useful_bin'] = train.apply (lambda row: label_race(row,'useful'), axis=1)
val['useful_bin'] = val.apply (lambda row: label_race(row,'useful'), axis=1)
test['useful_bin'] = test.apply (lambda row: label_race(row,'useful'), axis=1)

train.rename(columns={'useful_bin': 'label'}, inplace=True)
val.rename(columns={'useful_bin': 'label'}, inplace=True)
test.rename(columns={'useful_bin': 'label'}, inplace=True)

train['label'] = train['label'].fillna(0)
val['label']   = val['label'].fillna(0)
test['label']  = test['label'].fillna(0)

train = train.astype({"label":int})
val = val.astype({"label":int})
test = test.astype({"label":int})


# Create huggingface Dataset object

train_dataset = Dataset.from_pandas(train[['text','label']])
val_dataset   = Dataset.from_pandas(val[['text','label']])
test_dataset  = Dataset.from_pandas(test[['text','label']])


# Tokenize review text

from transformers import AlbertTokenizer, AlbertForSequenceClassification
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset_input = train_dataset.map(tokenize_function, batched=True)
test_dataset_input  = test_dataset.map(tokenize_function, batched=True)
val_dataset_input  = val_dataset.map(tokenize_function, batched=True)


# Retrieve pre-trained ALBERT from huggingface

model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=2)


# Load metrics

import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Set training arguments which are hyperparameters 

from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch",
                                  per_device_train_batch_size = 10, 
                                  per_device_eval_batch_size  = 10,
                                  learning_rate = 0.001)


# Start fine tuning model

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_input,
    eval_dataset=test_dataset_input,
    compute_metrics=compute_metrics,
)

trainer.train()


# Make predictions with fine tuned ALBERT

predictions = trainer.predict(test_dataset_input)
preds = np.argmax(predictions.predictions, axis=-1)

from datasets import load_metric

metric = load_metric("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)


# Save fine tuned model

trainer.save_model('/home/q/qshichao/BT4222/Archive/albert')