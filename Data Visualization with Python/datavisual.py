import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# create the dataframes using the json file..
# df = pd.read_json(r'./rain.json')
# print(df)

# print("df statistics: ", df.describe())


# df.plot(x='Month', y='Temperature')
# df.plot(x='Month', y='Rainfall')
# plt.show()

# create the dateframe using json file..

# plt.figure(figsize=(15, 5))
# plt.plot(df['Month'], df['Temperature'], label='Temperature')
# plt.show()


# plt.figure(figsize=(17,5))
# plt.plot( df['Month'], df['Rainfall'], label='Rainfall')
# plt.show()

# plt.plot( df['Month'], df['Rainfall'], label='Rainfall')
# plt.plot( df['Month'], df['Temperature'], label='Temperature')
# plt.legend()
# plt.show()

df = pd.read_csv(r'./tempYearly.csv')

print(df)

sns.set(rc={'figure.figsize': (12, 6)})
sns.scatterplot(x='Year', y='Temperature', data=df)
sns.regplot(x='Year', y='Temperature', data=df)
# plt.show()

data = pd.read_csv(r'./birthYearly.csv')
print(data)

dataP = data.pivot("month", "year", "births")
print(dataP)

sns.heatmap(dataP, annot=True, fmt='d')
# plt.show()

sns.jointplot(x="Rainfall", y="Temperature", data=df, kind="reg")
plt.show()
