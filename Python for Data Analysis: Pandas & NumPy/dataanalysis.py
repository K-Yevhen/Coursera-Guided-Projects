import numpy as np
# Let's define a one-dimensional array

list_1 = [50, 60, 80, 120, 200, 300, 500, 600]
print(list_1)

my_numpy_array = np.array(list_1)
print(my_numpy_array)
print(type(my_numpy_array))

my_matrix = np.array([ [2, 5, 8], [7, 3, 6] ])
print(my_matrix)

x = np.array( [[3, 7, 8, 3], [4, 3, 2, 2]] )
print(x)

x = np.random.rand(20)
print(x)

x = np.random.rand(3, 3)
print(x)

x = np.random.randint(1, 50)
print(x)

x = np.random.randint(1, 100, 15)
print(x)

x = np.arange(1, 50)
print(x)

x = np.ones((7, 7))
print(x)

x = np.zeros(8)
print(x)

x = int(input('Please enter a positive integer value: '))

x = np.random.randint(1, x, 10)
print(x)

# np.arange() returns an evenly spaced values within a given interval
x = np.arange(1, 10)
print(x)

y = np.arange(1, 10)
print(y)

# Add 2 numpy arrays together
sum = x + y
print(sum)

squared = x ** 2
print(squared)

sqrt = np.sqrt(squared)
print(sqrt)

z = np.exp(y)
print(z)

X = np.array([5, 7, 20])
Y = np.array([9, 15, 4])
Z = np.sqrt( X**2 + Y**2)

print(Z)

my_numpy_array = np.array([3, 5, 6, 2, 8, 10, 20, 50])
print(my_numpy_array)

print(my_numpy_array[0])
print(my_numpy_array[0:3])

my_numpy_array[0:4] = 7
print(my_numpy_array)


matrix = np.random.randint(1, 10, (4, 4))
print(matrix)


print(matrix[0])
print(matrix[0][3])

X = np.array([[2, 30, 20, -2, -4],
             [3, 4,  40, -3, -2],
             [-3, 4, -6, 90, 10],
             [25, 45, 34, 22, 12],
             [13, 24, 22, 32, 37]])
X[4] = 0
print(x)

matrix = np.random.randint(1, 10, (5, 5))
print(matrix)

new_matrix = matrix[ matrix > 7  ]
print(new_matrix)

new_matrix = matrix[ matrix % 2 == 1 ]
print(new_matrix)

X = np.array([[2, 30, 20, -2, -4],
    [3, 4, 40, -3, -2],
    [-3, 4, -6, 90, 10],
    [25, 45, 34, 22, 12],
    [13, 24, 22, 32, 37]])

X[ X < 0] = 0
X[ X % 2 == 1 ] = -2

print(X)

import pandas as pd

bank_client_df = pd.DataFrame({'Bank Client ID': [111, 222, 333, 444],
                              'Bank Client Name': ['Chanel', 'Steve', 'Mitch', 'Ryan'],
                              'Net Worth [$]': [3500, 29000, 10000, 2000],
                              'Years with bank': [3, 4, 9, 5]})
print(bank_client_df)

print(bank_client_df.head(2))
print(bank_client_df.tail(2))

portfolio_df = pd.DataFrame({'stock ticker symbol' :['AAPL',  'AMZN', 'T'],
                            'price per share[$]' :[3500, 200, 40],
                            'Number of stocks': [3, 4, 9]})
print(portfolio_df)

stocks_dollar_value = portfolio_df['price per share[$]'] * portfolio_df['Number of stocks']
print(stocks_dollar_value.sum())

import pandas as pd

house_price_df = pd.read_html('http://www.livingin-canada.com/house-prices-canada.html')
print(house_price_df)

house_price_df = pd.read_html('https://www.ssa.gov/oact/progdata/nra.html')
print(house_price_df[0])

bank_client_df = pd.DataFrame({'Bank Client ID': [111, 222, 333, 444],
                              'Bank Client Name': ['Chanel', 'Steve', 'Mitch', 'Ryan'],
                              'Net Worth [$]': [3500, 29000, 10000, 2000],
                              'Years with bank': [3, 4, 9, 5]})
print(bank_client_df)

df_loyal = bank_client_df[ bank_client_df['Years with bank'] >=5 ]
print(df_loyal)


del bank_client_df ['Bank Client ID']
print(bank_client_df)

df_loyal = bank_client_df[ bank_client_df['Net Worth [$]'] >=5000 ]
print(df_loyal)
print(df_loyal['Net Worth [$]'].sum())

bank_client_df = pd.DataFrame({'Bank client ID':[111, 222, 333, 444],
                               'Bank Client Name':['Chanel', 'Steve', 'Mitch', 'Ryan'],
                               'Net worth [$]':[3500, 29000, 10000, 2000],
                               'Years with bank':[3, 4, 9, 5]})
print(bank_client_df)

def networth_update(balance):
    return balance * 1.2

print(bank_client_df['Net worth [$]'].apply(networth_update))
print(bank_client_df['Bank Client Name'].apply(len))


def challenge(adding):
    return adding * 3 + 200
loya = bank_client_df['Net worth [$]'].apply(challenge)
print(loya)
print(loya.sum())

bank_client_df = pd.DataFrame({'Bank client ID':[111, 222, 333, 444],
                               'Bank Client Name':['Chanel', 'Steve', 'Mitch', 'Ryan'],
                               'Net worth [$]':[3500, 29000, 10000, 2000],
                               'Years with bank':[3, 4, 9, 5]})
print(bank_client_df)

print(bank_client_df.sort_values(by='Years with bank'))
print(bank_client_df.sort_values(by = 'Years with bank', inplace = True))
print(bank_client_df)

import pandas as pd

df1 = pd.DataFrame({ 'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3'],
                     'C': ['C0', 'C1', 'C2', 'C3'],
                     'D': ['D0', 'D1', 'D2', 'D3']},
                  index = [0, 1, 2, 3])
print(df1)

df2 = pd.DataFrame({ 'A': ['A4', 'A5', 'A6', 'A7'],
                     'B': ['B4', 'B5', 'B6', 'B7'],
                     'C': ['C4', 'C5', 'C6', 'C7'],
                     'D': ['D4', 'D5', 'D6', 'D7']},
                  index = [4, 5, 6, 7])
print(df2)

df3 = pd.DataFrame({ 'A': ['A8', 'A9', 'A10', 'A11'],
                     'B': ['B8', 'B9', 'B10', 'B11'],
                     'C': ['C8', 'C9', 'C10', 'C11'],
                     'D': ['D8', 'D9', 'D10', 'D11']},
                  index = [8, 9, 10, 11])
print(df3)

print(pd.concat([df1, df2, df3]))

raw_data = {'Bank Client ID': ['1', '2', '3', '4', '5'],
            'First Name': ['Nance', 'Alex', 'Shep', 'Max', 'Allen'],
            'Last Name': ['Rob', 'Ali', 'George', 'Mitch', 'Steve']}

Bank_df_1 = pd.DataFrame(raw_data, columns = ['Bank Client ID', 'First Name', 'Last Name'])
print(Bank_df_1)

raw_data = {'Bank Client ID': ['6', '7', '8', '9', '10'],
            'First Name': ['Bill', 'Dina', 'Sarah', 'Heather', 'Holly'],
            'Last Name': ['Christian', 'Mo', 'Steve', 'Rob', 'Michelle']}
Bank_df_2 = pd.DataFrame(raw_data, columns = ['Bank Client ID', 'First Name', 'Last Name'])
print(Bank_df_2)

raw_data = { 'Bank Client ID' : ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
             'Annual Salary [$/year]': [25000, 35000, 45000, 48000, 49000, 32000, 33000, 34000, 23000, 22000]}

bank_df_salary = pd.DataFrame(raw_data, columns = ['Bank Client ID', 'Annual Salary [$/year]'])
print(bank_df_salary)

bank_df_all = pd.concat([Bank_df_1, Bank_df_2])
print(bank_df_all)

bank_df_all = pd.merge(bank_df_all, bank_df_salary, on = 'Bank Client ID')
print(bank_df_all)

new_client = {'Bank Client ID': ['11'],
             'First Name': ['Ryan'],
             'Last Name': ['Ahmed'],
             'Annual Salary [$/year]': [5000]}

new_client_df = pd.DataFrame(new_client, columns = ['Bank Client ID', 'First Name', 'Last Name', 'Annual Salary [$/year]'])

print(new_client_df)

new_df = pd.concat([bank_df_all, new_client_df], axis = 0)
print(new_df)
