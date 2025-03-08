import pandas as pd

df =  pd.read_csv('test.csv')

df['Concreteness'].replace(0, None, inplace=True)
df['Familiarity'].replace(0, None, inplace=True)

df.to_csv('test.csv')

# Count missing values per column
missing_values = df.isnull().sum()

print("Number of missing values in each column:")
print(missing_values)

print(len(df))