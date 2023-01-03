import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# print('Modules are imported.')

corona_dataset_csv = pd.read_csv("datasets/covid19_Confirmed_dataset.csv")
# print(corona_dataset_csv.head(10))
# print(corona_dataset_csv.shape)

# corona_dataset_csv.drop(["Lat", "Long"], axis=1, inplace=True)
# print(corona_dataset_csv.head(10))

corona_dataset_aggregated = corona_dataset_csv.groupby("Country/Region").sum()
# print(corona_dataset_aggregated.head())

# corona_dataset_aggregated.loc["China"].plot()
# corona_dataset_aggregated.loc["Italy"].plot()
# corona_dataset_aggregated.loc["Spain"].plot()
# plt.legend()
# plt.suptitle("Covid-19")
# plt.title("Aggregated Data Set")
# xplt.show()

# corona_dataset_aggregated.loc['China'][:30].plot()
# corona_dataset_aggregated.loc['China'].diff().plot()
# plt.show()
# print(corona_dataset_aggregated.loc['China'].diff().max())
# print(corona_dataset_aggregated.loc['US'].diff().max())

countries = list(corona_dataset_aggregated.index)
max_infection_rates = []
for c in countries:
    max_infection_rates.append(corona_dataset_aggregated.loc[c].diff().max())

# print(max_infection_rates)

corona_dataset_aggregated["max_infection_rate"] = max_infection_rates
# print(corona_dataset_aggregated.head())

corona_data = pd.DataFrame(corona_dataset_aggregated["max_infection_rate"])
# print(corona_data.head())
# corona_data.plot()
# plt.show()

happiness_report_csv = pd.read_csv('datasets/worldwide_happiness_report.csv')
useless_cols = ["Overall rank", "Score", "Generosity", "Perceptions of corruption"]
happiness_report_csv.drop(useless_cols, axis=1, inplace=True)
# print(happiness_report_csv.head())

happiness_report_csv.set_index("Country or region", inplace=True)
# print(happiness_report_csv.head())

# print(corona_data.head())
# print(corona_data.shape)

# print(happiness_report_csv.head())
# print(happiness_report_csv.shape)

data = corona_data.join(happiness_report_csv, how="inner")
# print(data.head())
# print(data.corr())

x = data["GDP per capita"]
y = data["max_infection_rate"]
sns.scatterplot(data=data, x=data["GDP per capita"], y=data["max_infection_rate"])
plt.title('Plotting GDP vs maximum Infection rate')
plt.suptitle("Task 5.1")
plt.show()

sns.regplot(x, np.log(y))
plt.show()