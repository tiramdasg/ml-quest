# necessary libraries for ml preprocessing
import pandas as pd
import matplotlib as plt
import numpy as np

dataset = pd.read_csv('Data.csv')
# features or independent variable are columns = X
# dependent variable is the target variable - usually last column = y
X = dataset.iloc[: , :-1].values
y = dataset.iloc[:, -1].values
# '.values' so it is not in Series format as 'dataset' is a dataframe
# print(X, '\n', y)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

# Print the number of missing entries in each column
missing_entries = dataset.isnull().sum()
print(missing_entries)
