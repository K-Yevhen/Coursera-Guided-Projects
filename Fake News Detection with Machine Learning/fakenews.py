# !pip install plotly
# !pip install --upgrade nbformat
# !pip install nltk
# !pip install spacy # spaCy is an open-source software library for advanced natural language processing
# !pip install WordCloud
# !pip install gensim # Gensim is an open-source library for unsupervised topic modeling and natural language processing
# import nltk
# nltk.download('punkt')

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras_preprocessing.sequence import pad_sequences
from wordcloud import WordCloud, STOPWORDS
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
# import keras
from keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.python.keras.models import Model
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)
# setting the style of the notebook to be monokai theme
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them.

df_true = pd.read_csv('True.csv')
df_fake = pd.read_csv('Fake.csv')

df_true.isnull().sum()
df_fake.isnull().sum()

# print(df_fake.info())
# print(df_true.info())

df_true['isfake'] = 0
# print(df_true.head())

df_fake['isfake'] = 1
# print(df_fake.head())

df = pd.concat([df_true, df_fake]).reset_index(drop=True)
# print(df)

df.drop(columns = ['date'], inplace=True)

df['original'] = df['title'] + ' ' + df['text']
# print(df.head())

# print(df['original'][0])

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# print(stop_words)


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)

    return result

df['clean'] = df['original'].apply(preprocess)
# print(df['original'][0])
# print(print(df['clean'][0]))
list_of_words = []
for i in df.clean:
    for j in i:
        list_of_words.append(j)
# print(list_of_words)

# print(len(list_of_words))
total_words = len(list(set(list_of_words)))
# print(total_words)

df['clean_joined'] = df['clean'].apply(lambda x: " ".join(x))
# print(df)

# plt.figure(figsize=(8, 8))
# sns.countplot(y="subject", data=df)

# plt.figure(figsize = (20,20))
# wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = stop_words).generate(" ".join(df[df.isfake == 1].clean_joined))
# plt.imshow(wc, interpolation = 'bilinear')

# plt.figure(figsize = (20,20))
# wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = stop_words).generate(" ".join(df[df.isfake == 0].clean_joined))
# plt.imshow(wc, interpolation = 'bilinear')

maxlen = -1
for doc in df.clean_joined:
    tokens = nltk.word_tokenize(doc)
    if(maxlen<len(tokens)):
        maxlen = len(tokens)
# print("The maximum number of words in any document is =", maxlen)

import plotly.express as px
fig = px.histogram(x = [len(nltk.word_tokenize(x)) for x in df.clean_joined], nbins = 100)
# fig.show()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.isfake, test_size = 0.2)

from nltk import word_tokenize
# Create a tokenizer to tokenize the words and create sequences of tokenized words

tokenizer = Tokenizer(num_words = total_words)
tokenizer.fit_on_texts(x_train)
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

# print("The encoding for document\n",df.clean_joined[0],"\n is : ",train_sequences[0])

padded_train = pad_sequences(train_sequences,maxlen = 40, padding = 'post', truncating = 'post')
padded_test = pad_sequences(test_sequences,maxlen = 40, truncating = 'post')

# for i,doc in enumerate(padded_train[:2]):
#     print("The padded encoding for document",i+1," is : ",doc)


# Sequential Model
model = Sequential()

# embeddidng layer
model.add(Embedding(total_words, output_dim = 128))
# model.add(Embedding(total_words, output_dim = 240))


# Bi-Directional RNN and LSTM
# model.add(Bidirectional(LSTM(128)))

# Dense layers
# model.add(Dense(128, activation = 'relu'))
# model.add(Dense(1,activation= 'sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# model.summary()

y_train = np.asarray(y_train)
# model.fit(padded_train, y_train, batch_size = 64, validation_split = 0.1, epochs = 2)

pred = model.predict(padded_test)

prediction = []
for i in range(len(pred)):
    if pred[i].item() > 0.5:
        prediction.append(1)
    else:
        prediction.append(0)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(list(y_test), prediction)
# print("Model Accuracy : ", accuracy)

# get the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(list(y_test), prediction)
plt.figure(figsize = (25, 25))
sns.heatmap(cm, annot = True)

category = { 0: 'Fake News', 1 : "Real News"}
