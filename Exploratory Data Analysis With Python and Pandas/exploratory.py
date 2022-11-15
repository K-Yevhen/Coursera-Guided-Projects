import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calmap
from pandas_profiling import ProfileReport

df = pd.read_csv('supermarket_sales.csv')
print(df.head())

print(df.columns)
print(df.dtypes)

df['Date'] = pd.to_datetime(df['Date'])

print(df.set_index('Date',inplace=True))
print(df.describe())
sns.distplot(df['Rating'])
sns.distplot(df['Rating'])
plt.axvline(x=np.mean(df['Rating']), c='red', ls='--', label='mean')
plt.axvline(x=np.mean(df['Rating']), c='red', ls='--', label='mean')
plt.axvline(x=np.mean(df['Rating']), c='red', ls='--', label='mean')
plt.legend()
print(df.columns.tolist())

print(df.hist(figsize=(10,10)))

sns.countplt(df['Branch'])

print(df['Branch'].value_counts())

print(sns.countplot(df['Rating']))

print(sns.scatterplot(df['Rating'], df['gross income']))

sns.regplot(df['Rating'], df['gross income'])

sns.boxplot(x=df['Gender'], y=df['gross income'])

sns.lineplot(x=df.groupby(df.index).mean().index, y=df.groupby(df.index).mean()['gross income'])

df.duplicated().sum()
df[df.duplicated()==True]
df.drop_duplicates(inplace=True)
df.isna().sum()
sns.heatmap(df.isnull())
sns.heatmap(df.isnull(),cbar=False)
df.fillna(df.mean(), inplace=True)

df.mode().iloc[0]

dataset = pd.read_csv('supermarket_sales.csv')
prof = ProfileReport(dataset)
print(prof)

round(np.corrcoef(df['gross income'], df['Rating'])[1][0],2)
np.round(df.corr(),2)
sns.heatmap(np.round(df.corr(),2),annot=True)
