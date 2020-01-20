#!/usr/bin/env python
# coding: utf-8

# """
# # Final Project: Quora Insincere Questions Classification (Competition available on Kaggle)
# 
# About Quora and the dataset:
#     An existential problem for any major website today is how to handle toxic and divisive content. 
#     Quora wants to tackle this problem head-on to keep their platform a place where users can feel 
#     safe sharing their knowledge with the world.
# 
#     Quora is a platform that empowers people to learn from each other. 
#     On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. 
#     A key challenge is to weed out insincere questions -- those founded upon false premises, 
#     or that intend to make a statement rather than look for helpful answers.    
# 
# Description and Problem Statement:
#     An insincere question is defined as a question intended to make a statement rather than look for helpful answers. Some characteristics that can signify that a question is insincere:
# 
#         Has a non-neutral tone
#             Has an exaggerated tone to underscore a point about a group of people
#             Is rhetorical and meant to imply a statement about a group of people
#         Is disparaging or inflammatory
#             Suggests a discriminatory idea against a protected class of people, or seeks confirmation of a stereotype
#             Makes disparaging attacks/insults against a specific person or group of people
#             Based on an outlandish premise about a group of people
#             Disparages against a characteristic that is not fixable and not measurable
#         Isn't grounded in reality
#             Based on false information, or contains absurd assumptions
#         Uses sexual content (incest, bestiality, pedophilia) for shock value, and not to seek genuine answers
#     
#     The training data includes the question that was asked, and whether it was identified as insincere (target = 1). 
#     The ground-truth labels contain some amount of noise: they are not guaranteed to be perfect.
#     
#     File descriptions
#         train.csv - the training set
#         test.csv - the test set
#         sample_submission.csv - A sample submission in the correct format
#         embeddings -
#                 GoogleNews-vectors-negative300 - https://code.google.com/archive/p/word2vec/
#                 glove.840B.300d - https://nlp.stanford.edu/projects/glove/
#                 paragram_300_sl999 - https://cogcomp.org/page/resource_view/106
#                 wiki-news-300d-1M - https://fasttext.cc/docs/en/english-vectors.html
# 
#     Data fields
#         qid - unique question identifier
#         question_text - Quora question text
#         target - a question labeled "insincere" has a value of 1, otherwise 0
#   
# Performance Metric Used: F1-score
# """

import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
 
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, CuDNNLSTM
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
 
 #Below for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)

#Plot --> count Sincere and Insincere questions in Quora Train Dataset
train_df.target.value_counts().plot(kind='bar', color='Blue', title='Frequency of Sincere vs Insincere Questions', x='0-Sincere or 1-Insincere Question', y='Number of Questions')

#Pie Distribution Plot using Plotly
labels = (np.array((train_df['target'].value_counts()).index))
values = (np.array(((train_df['target'].value_counts()) / (train_df['target'].value_counts()).sum())*100))

trace = go.Pie(labels=labels, values=values)
layout = go.Layout(
    title='Target distribution',
    font=dict(size=18),
    width=600,
    height=600,
)
data=[trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="usertype.html")

#For simplicity
sincere_questions = train_df[train_df['target'] == 0]
insincere_questions = train_df[train_df['target'] == 1]

# Sincere Word Cloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

words_sincere = ' '.join(sincere_questions['question_text'])
wordcloud_sincere = WordCloud(stopwords=STOPWORDS,max_words=500,
                      background_color='white',min_font_size=6,
                      width=3000,collocations=False,
                      height=2500
                     ).generate(words_sincere)
plt.figure(1,figsize=(16, 16))
plt.imshow(wordcloud_sincere)
plt.axis('off')
plt.title("Word Cloud of Sincere Train Questions")
plt.show()

# Insincere Word Cloud
words_insincere = ' '.join(insincere_questions['question_text'])
wordcloud_insincere = WordCloud(stopwords=STOPWORDS,max_words=500,
                      background_color='white',min_font_size=6,
                      width=3000,collocations=False,
                      height=2500
                     ).generate(words_insincere)
plt.figure(1,figsize=(16, 16))
plt.imshow(wordcloud_insincere)
plt.axis('off')
plt.title("Word Cloud of Insincere Train Questions")
plt.show()

#  * Split the training dataset into training and validation
#  * Fill up the missing values with '_na_'
#  * Tokenize the text column and convert them to vector sequences
#  * Pad the sequence as needed - if the number of words in the text is greater than 'max_len' trunacate them to 'max_len' or if the number of words in the text is lesser than 'max_len' add zeros for remaining values.

## split to train and val
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)

## some config values 
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use

## fill up the missing values
train_X = train_df["question_text"].fillna("_na_").values
val_X = val_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values

# **1. Without Pretrained Embeddings - Bidirectional GRU model:**
#  
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# Train the model using train sample and monitor the metric on the validation sample. Model runs for 2 epochs.

## Train the model 
history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))

# Now let us get the validation sample predictions and also get the best threshold for F1 score. 
pred_noemb_val_y_gru = model.predict([val_X], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("At threshold {0}, F1 score is {1:.4f}, Precision is {2:.4f} and Recall is {3:.4f}".format(thresh, metrics.f1_score(val_y, (pred_noemb_val_y_gru>thresh).astype(int)), metrics.precision_score(val_y, (pred_noemb_val_y_gru>thresh).astype(int)),metrics.recall_score(val_y, (pred_noemb_val_y_gru>thresh).astype(int))))

# ##### Plot of F1 score, Precision and Recall for different thresholds 
f1s = []
precisions = []
recalls = []
    
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    f1 = metrics.f1_score(val_y, (pred_noemb_val_y_gru>thresh).astype(int))
    precision = metrics.precision_score(val_y, (pred_noemb_val_y_gru>thresh).astype(int))
    recall = metrics.recall_score(val_y, (pred_noemb_val_y_gru>thresh).astype(int))

    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)

    data = pd.DataFrame(data = {
        'F1': f1s,
        'Precision': precisions,
        'Recall': recalls})
sns.lineplot(data=data, palette='muted', linewidth=2.5, dashes=False)
sns.despine()
plt.show()

# Plot of Train/Test loss v/s Epoch
data = pd.DataFrame(data={'Train': history.history['loss'], 'Test': history.history['val_loss']})
ax = sns.lineplot(data=data, palette="pastel", linewidth=2.5, dashes=False)
ax.set(xlabel='Epoch', ylabel='Loss', title='Loss')
sns.despine()
plt.show()

# Get the test set predictions and save 
pred_noemb_test_y_gru = model.predict([test_X], batch_size=1024, verbose=1)

# Clean up some memory before going to the next step.
del model, inp, x, history, data, ax
import gc; gc.collect()
time.sleep(10)

# **2. Without Pretrained Embeddings - Bidirectional LSTM model:**
# 
# Now second we are training a Bidirectional LSTM model. We will not use any pre-trained word embeddings for this model and the embeddings will be learnt from scratch.
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)
x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# Train the model using train sample and monitor the metric on the valid sample. Model running for 2 epochs.

## Train the model 
history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))

# Now let us get the validation sample predictions and also get the best threshold for F1 score. 
pred_noemb_val_y_lstm = model.predict([val_X], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("At threshold {0}, F1 score is {1}, Precision is {2} and Recall is {3}".format(thresh, metrics.f1_score(val_y, (pred_noemb_val_y_lstm>thresh).astype(int)), metrics.precision_score(val_y, (pred_noemb_val_y_lstm>thresh).astype(int)),metrics.recall_score(val_y, (pred_noemb_val_y_lstm>thresh).astype(int))))

# Plot the F1 score, Precision and Recall for different thresholds 
f1s = []
precisions = []
recalls = []
    
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    f1 = metrics.f1_score(val_y, (pred_noemb_val_y_lstm>thresh).astype(int))
    precision = metrics.precision_score(val_y, (pred_noemb_val_y_lstm>thresh).astype(int))
    recall = metrics.recall_score(val_y, (pred_noemb_val_y_lstm>thresh).astype(int))

    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)

    data = pd.DataFrame(data = {
        'F1': f1s,
        'Precision': precisions,
        'Recall': recalls})
sns.lineplot(data=data, palette='muted', linewidth=2.5, dashes=False)
sns.despine()
plt.show()

# Plot of Train/Test loss v/s Epoch
data = pd.DataFrame(data={'Train': history.history['loss'], 'Test': history.history['val_loss']})
ax = sns.lineplot(data=data, palette="pastel", linewidth=2.5, dashes=False)
ax.set(xlabel='Epoch', ylabel='Loss', title='Loss')
sns.despine()
plt.show()

# Get the test set predictions aand save
pred_noemb_test_y_lstm = model.predict([test_X], batch_size=1024, verbose=1)

# Clean up some memory before we going to the next step.
del model, inp, x, history, data, ax
import gc; gc.collect()
time.sleep(10)

# So we got some baseline GRU and LSTM model without pre-trained embeddings. Now let us use the provided embeddings and rebuild the model again to see the performance. 
# 
# 

# We have four different types of embeddings.
#  * GoogleNews-vectors-negative300 - https://code.google.com/archive/p/word2vec/
#  * glove.840B.300d - https://nlp.stanford.edu/projects/glove/
#  * paragram_300_sl999 - https://cogcomp.org/page/resource_view/106
#  * wiki-news-300d-1M - https://fasttext.cc/docs/en/english-vectors.html
#  
# 
# 
# 
# **3. Glove Embeddings - Bidirectional GRU model:**
# 
# Using Glove embeddings and rebuilding the GRU model.
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))

pred_glove_val_y_gru = model.predict([val_X], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("At threshold {0}, F1 score is {1}, Precision is {2} and Recall is {3}".format(thresh, metrics.f1_score(val_y, (pred_glove_val_y_gru>thresh).astype(int)), metrics.precision_score(val_y, (pred_glove_val_y_gru>thresh).astype(int)),metrics.recall_score(val_y, (pred_glove_val_y_gru>thresh).astype(int))))

# Plot of F1 score, Precision and Recall for different thresholds 
f1s = []
precisions = []
recalls = []
    
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    f1 = metrics.f1_score(val_y, (pred_glove_val_y_gru>thresh).astype(int))
    precision = metrics.precision_score(val_y, (pred_glove_val_y_gru>thresh).astype(int))
    recall = metrics.recall_score(val_y, (pred_glove_val_y_gru>thresh).astype(int))

    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)

    data = pd.DataFrame(data = {
        'F1': f1s,
        'Precision': precisions,
        'Recall': recalls})
sns.lineplot(data=data, palette='muted', linewidth=2.5, dashes=False)
sns.despine()
plt.show()

# Plot of Train/Test loss v/s Epoch
data = pd.DataFrame(data={'Train': history.history['loss'], 'Test': history.history['val_loss']})
ax = sns.lineplot(data=data, palette="pastel", linewidth=2.5, dashes=False)
ax.set(xlabel='Epoch', ylabel='Loss', title='Loss')
sns.despine()
plt.show()

# Results are better than the model without pretrained embeddings.
pred_glove_test_y_gru = model.predict([test_X], batch_size=1024, verbose=1)

# Clean up memory
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x, history, data, ax
import gc; gc.collect()
time.sleep(10)

# **4. Glove Embeddings - Bidirectional LSTM model:**
# 
# Using the Glove embeddings and rebuilding the LSTM model.
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))

pred_glove_val_y_lstm = model.predict([val_X], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("At threshold {0}, F1 score is {1}, Precision is {2} and Recall is {3}".format(thresh, metrics.f1_score(val_y, (pred_glove_val_y_lstm>thresh).astype(int)), metrics.precision_score(val_y, (pred_glove_val_y_lstm>thresh).astype(int)),metrics.recall_score(val_y, (pred_glove_val_y_lstm>thresh).astype(int))))

# Plot of F1 score, Precision and Recall for different thresholds 
f1s = []
precisions = []
recalls = []
    
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    f1 = metrics.f1_score(val_y, (pred_glove_val_y_lstm>thresh).astype(int))
    precision = metrics.precision_score(val_y, (pred_glove_val_y_lstm>thresh).astype(int))
    recall = metrics.recall_score(val_y, (pred_glove_val_y_lstm>thresh).astype(int))

    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)

    data = pd.DataFrame(data = {
        'F1': f1s,
        'Precision': precisions,
        'Recall': recalls})
sns.lineplot(data=data, palette='muted', linewidth=2.5, dashes=False)
sns.despine()
plt.show()

# Plot of Train/Test loss v/s Epoch
data = pd.DataFrame(data={'Train': history.history['loss'], 'Test': history.history['val_loss']})
ax = sns.lineplot(data=data, palette="pastel", linewidth=2.5, dashes=False)
ax.set(xlabel='Epoch', ylabel='Loss', title='Loss')
sns.despine()
plt.show()

# Results are better than the model without pretrained embeddings.
pred_glove_test_y_lstm = model.predict([test_X], batch_size=1024, verbose=1)

# Clean up memory
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x, history, data, ax
import gc; gc.collect()
time.sleep(10)

# **5. Wiki News FastText Embeddings - Bidirectional GRU model:**
# 
# Using the FastText embeddings trained on Wiki News corpus in place of Glove embeddings and rebuilding the GRU model.
EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))

pred_fasttext_val_y_gru = model.predict([val_X], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("At threshold {0}, F1 score is {1}, Precision is {2} and Recall is {3}".format(thresh, metrics.f1_score(val_y, (pred_fasttext_val_y_gru>thresh).astype(int)), metrics.precision_score(val_y, (pred_fasttext_val_y_gru>thresh).astype(int)),metrics.recall_score(val_y, (pred_fasttext_val_y_gru>thresh).astype(int))))

# Plot of F1 score, Precision and Recall for different thresholds 
f1s = []
precisions = []
recalls = []
    
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    f1 = metrics.f1_score(val_y, (pred_fasttext_val_y_gru>thresh).astype(int))
    precision = metrics.precision_score(val_y, (pred_fasttext_val_y_gru>thresh).astype(int))
    recall = metrics.recall_score(val_y, (pred_fasttext_val_y_gru>thresh).astype(int))

    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)

    data = pd.DataFrame(data = {
        'F1': f1s,
        'Precision': precisions,
        'Recall': recalls})
sns.lineplot(data=data, palette='muted', linewidth=2.5, dashes=False)
sns.despine()
plt.show()

# Plot of Train/Test loss v/s Epoch
data = pd.DataFrame(data={'Train': history.history['loss'], 'Test': history.history['val_loss']})
ax = sns.lineplot(data=data, palette="pastel", linewidth=2.5, dashes=False)
ax.set(xlabel='Epoch', ylabel='Loss', title='Loss')
sns.despine()
plt.show()

pred_fasttext_test_y_gru = model.predict([test_X], batch_size=1024, verbose=1)

# Cleanup memory
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x, history, data, ax
import gc; gc.collect()
time.sleep(10)

# **6. Wiki News FastText Embeddings - Bidirectional LSTM model:**
# 
# Using the FastText embeddings trained on Wiki News corpus in place of Glove embeddings and rebuilding the LSTM model.
EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))

pred_fasttext_val_y_lstm = model.predict([val_X], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("At threshold {0}, F1 score is {1}, Precision is {2} and Recall is {3}".format(thresh, metrics.f1_score(val_y, (pred_fasttext_val_y_lstm>thresh).astype(int)), metrics.precision_score(val_y, (pred_fasttext_val_y_lstm>thresh).astype(int)),metrics.recall_score(val_y, (pred_fasttext_val_y_lstm>thresh).astype(int))))

# Plot of F1 score, Precision and Recall for different thresholds 
f1s = []
precisions = []
recalls = []
    
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    f1 = metrics.f1_score(val_y, (pred_fasttext_val_y_lstm>thresh).astype(int))
    precision = metrics.precision_score(val_y, (pred_fasttext_val_y_lstm>thresh).astype(int))
    recall = metrics.recall_score(val_y, (pred_fasttext_val_y_lstm>thresh).astype(int))

    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)

    data = pd.DataFrame(data = {
        'F1': f1s,
        'Precision': precisions,
        'Recall': recalls})
sns.lineplot(data=data, palette='muted', linewidth=2.5, dashes=False)
sns.despine()
plt.show()

# Plot of Train/Test loss v/s Epoch
data = pd.DataFrame(data={'Train': history.history['loss'], 'Test': history.history['val_loss']})
ax = sns.lineplot(data=data, palette="pastel", linewidth=2.5, dashes=False)
ax.set(xlabel='Epoch', ylabel='Loss', title='Loss')
sns.despine()
plt.show()

pred_fasttext_test_y_lstm = model.predict([test_X], batch_size=1024, verbose=1)

# Cleanup memory
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x, history, data, ax
import gc; gc.collect()
time.sleep(10)

# **7. Paragram Embeddings - Bidirectional GRU model:**
# 
# Using the paragram embeddings and rebuilding the model and make predictions.
EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))

pred_paragram_val_y_gru = model.predict([val_X], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("At threshold {0}, F1 score is {1}, Precision is {2} and Recall is {3}".format(thresh, metrics.f1_score(val_y, (pred_paragram_val_y_gru>thresh).astype(int)), metrics.precision_score(val_y, (pred_paragram_val_y_gru>thresh).astype(int)),metrics.recall_score(val_y, (pred_paragram_val_y_gru>thresh).astype(int))))

# Plot of F1 score, Precision and Recall for different thresholds 
f1s = []
precisions = []
recalls = []
    
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    f1 = metrics.f1_score(val_y, (pred_paragram_val_y_gru>thresh).astype(int))
    precision = metrics.precision_score(val_y, (pred_paragram_val_y_gru>thresh).astype(int))
    recall = metrics.recall_score(val_y, (pred_paragram_val_y_gru>thresh).astype(int))

    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)

    data = pd.DataFrame(data = {
        'F1': f1s,
        'Precision': precisions,
        'Recall': recalls})
sns.lineplot(data=data, palette='muted', linewidth=2.5, dashes=False)
sns.despine()
plt.show()

# Plot of Train/Test loss v/s Epoch
data = pd.DataFrame(data={'Train': history.history['loss'], 'Test': history.history['val_loss']})
ax = sns.lineplot(data=data, palette="pastel", linewidth=2.5, dashes=False)
ax.set(xlabel='Epoch', ylabel='Loss', title='Loss')
sns.despine()
plt.show()

pred_paragram_test_y_gru = model.predict([test_X], batch_size=1024, verbose=1)

# Clean up memory
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x, history, data, ax
import gc; gc.collect()
time.sleep(10)

# **8. Paragram Embeddings - Bidirectional LSTM model:**
# 
# Using the paragram embeddings and build the model and make predictions.
EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))

pred_paragram_val_y_lstm = model.predict([val_X], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("At threshold {0}, F1 score is {1}, Precision is {2} and Recall is {3}".format(thresh, metrics.f1_score(val_y, (pred_paragram_val_y_lstm>thresh).astype(int)), metrics.precision_score(val_y, (pred_paragram_val_y_lstm>thresh).astype(int)),metrics.recall_score(val_y, (pred_paragram_val_y_lstm>thresh).astype(int))))

# Plot of F1 score, Precision and Recall for different thresholds 
f1s = []
precisions = []
recalls = []
    
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    f1 = metrics.f1_score(val_y, (pred_paragram_val_y_lstm>thresh).astype(int))
    precision = metrics.precision_score(val_y, (pred_paragram_val_y_lstm>thresh).astype(int))
    recall = metrics.recall_score(val_y, (pred_paragram_val_y_lstm>thresh).astype(int))

    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)

    data = pd.DataFrame(data = {
        'F1': f1s,
        'Precision': precisions,
        'Recall': recalls})
sns.lineplot(data=data, palette='muted', linewidth=2.5, dashes=False)
sns.despine()
plt.show()

# Plot of Train/Test loss v/s Epoch
data = pd.DataFrame(data={'Train': history.history['loss'], 'Test': history.history['val_loss']})
ax = sns.lineplot(data=data, palette="pastel", linewidth=2.5, dashes=False)
ax.set(xlabel='Epoch', ylabel='Loss', title='Loss')
sns.despine()
plt.show()

pred_paragram_test_y_lstm = model.predict([test_X], batch_size=1024, verbose=1)

# Clean up memory
del word_index, embeddings_index, all_embs, embedding_matrix, model, inp, x, history, data, ax
import gc; gc.collect()
time.sleep(10)

# **Final Model:**
# 
# Though the results of the models with different pre-trained embeddings are similar, there is a good chance that they might capture different type of information from the data. So averaging of predictions from these three models.

# **Averaging all 3 embeddings - Bidirectional GRU Model**
pred_val_y_gru = 0.34*pred_glove_val_y_gru + 0.33*pred_fasttext_val_y_gru + 0.33*pred_paragram_val_y_gru 
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("At threshold {0}, F1 score is {1}, Precision is {2} and Recall is {3}".format(thresh, metrics.f1_score(val_y, (pred_val_y_gru>thresh).astype(int)), metrics.precision_score(val_y, (pred_val_y_gru>thresh).astype(int)),metrics.recall_score(val_y, (pred_val_y_gru>thresh).astype(int))))

# Plot of F1 score, Precision and Recall for different thresholds 
f1s = []
precisions = []
recalls = []
    
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    f1 = metrics.f1_score(val_y, (pred_val_y_gru>thresh).astype(int))
    precision = metrics.precision_score(val_y, (pred_val_y_gru>thresh).astype(int))
    recall = metrics.recall_score(val_y, (pred_val_y_gru>thresh).astype(int))

    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)

    data = pd.DataFrame(data = {
        'F1': f1s,
        'Precision': precisions,
        'Recall': recalls})
sns.lineplot(data=data, palette='muted', linewidth=2.5, dashes=False)
sns.despine()
plt.show()

# **Averaging all 3 embeddings - Bidirectional LSTM Model**
pred_val_y_lstm = 0.34*pred_glove_val_y_lstm + 0.33*pred_fasttext_val_y_lstm + 0.33*pred_paragram_val_y_lstm 
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("At threshold {0}, F1 score is {1}, Precision is {2} and Recall is {3}".format(thresh, metrics.f1_score(val_y, (pred_val_y_lstm>thresh).astype(int)), metrics.precision_score(val_y, (pred_val_y_lstm>thresh).astype(int)),metrics.recall_score(val_y, (pred_val_y_lstm>thresh).astype(int))))

# Plot of F1 score, Precision and Recall for different thresholds 
f1s = []
precisions = []
recalls = []
    
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    f1 = metrics.f1_score(val_y, (pred_val_y_lstm>thresh).astype(int))
    precision = metrics.precision_score(val_y, (pred_val_y_lstm>thresh).astype(int))
    recall = metrics.recall_score(val_y, (pred_val_y_lstm>thresh).astype(int))

    f1s.append(f1)
    precisions.append(precision)
    recalls.append(recall)

    data = pd.DataFrame(data = {
        'F1': f1s,
        'Precision': precisions,
        'Recall': recalls})
sns.lineplot(data=data, palette='muted', linewidth=2.5, dashes=False)
sns.despine()
plt.show()

# Although we see that both the aggregated results from the models are performing equally good, we have created the final submission file using the GRU model as the final f1 scores are 0.683 for LSTM and 0.684 for GRU
pred_test_y_gru = 0.34*pred_glove_test_y_gru + 0.33*pred_fasttext_test_y_gru + 0.33*pred_paragram_test_y_gru
pred_test_y_gru = (pred_test_y_gru>0.35).astype(int)
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y_gru
out_df.to_csv("submission.csv", index=False)