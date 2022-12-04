import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_columns = None
import seaborn as sns
# %matplotlib inline
plt.style.use('ggplot')
import datetime
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, KFold
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import nltk
nltk.download('stopwords')
stop = set(stopwords.words('english'))
import os
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import json
import ast
from urllib.request import urlopen
from PIL import Image
import eli5
from sklearn.linear_model import LinearRegression

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

# print(train.head()) - showing 5 first data
# print(test.tail()) - showing 5 last data


# train.revenue.hist() - visualisation of revenue data
# fig, ax = plt.subplots(figsize=(16, 6))
# plt.subplot(1, 2, 1)
# sns.histplot(train['revenue'], kde=False)
# plt.title('Distribution of revenue'); - setting the title
# plt.subplot(1, 2, 2)
# sns.histplot(np.log1p(train['revenue']), kde=False)
# plt.title('Distribution of lof-transformed revenue'); - setting the title
# plt.show() - showing the figure

train['log_revenue'] = np.log1p(train['revenue'])

# plt.figure(figsize=(16, 8))
# plt.subplot(1, 2, 1)
# sns.scatterplot(data=train, x='budget', y='revenue')
# sns.scatterplot(data=train, y='revenue')
# plt.txubplot(1, 2, 2)
# sns.scatterplot(data=train, x=np.log1p(train['budget']), y=train['log_revenue'])
# plt.title('Log Revenue vs Log Budget')
# plt.show()

train['log_budget'] = np.log1p(train['budget'])
test['log_budget'] = np.log1p(train['budget'])

# print(train['homepage'].value_counts().head(10))

train['has_homepage'] = 0
train.loc[train['homepage'].isnull() == False, 'has_homepage'] = 1

# sns.catplot(x='has_homepage', y='revenue', data=train)
# plt.title('Revenue for films with and without a homepage')
# plt.show()

language_data = train.loc[train['original_language'].isin(train['original_language'].value_counts().head(10).index)]

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
sns.boxplot(x='original_language', y='revenue', data=language_data)
plt.title('Mean revenue per language')
plt.subplot(1, 2, 2)
sns.boxplot(x='original_language', y='log_revenue', data=language_data)
plt.title('Mean log revenue per language')
plt.show()

plt.figure(figsize=(12, 12))
text = ' '.join(train['original_title'].values)
wordcloud = WordCloud(max_font_size=None,
                    background_color='white',                     width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top words across movie title')
plt.axis('off')
plt.show()

plt.figure(figsize=(12, 12))
text = ' '.join(train['overview'].fillna('').values)
wordcloud = WordCloud(max_font_size=None,
                      background_color='white',
                      width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top words across movie overview');
plt.axis('off');
plt.show()


# vectorizer = TfidfVectorizer(
#     sublinear_tf=True,
#     analyzer='word',
#     token_pattern=r'\w{1,}',
#     ngram_range=(1, 2),
#     min_df=5
# )


# overview_text = vectorizer.fit_transform(train['overview'].fillna(''))
# linreg = LinearRegression()
# linreg.fit(overview_text, train['log_revenue'])
# eli5.show_weights(linreg, vec=vectorizer, top=20, feature_filter=lambda x: x != '<BIAS>')
