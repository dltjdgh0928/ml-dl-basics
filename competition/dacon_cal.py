import pandas as pd


path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

print(train_csv.shape)
print(test_csv.shape)
print(train_csv.isna().sum())
print(test_csv.isna().sum())
print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())
print(test_csv.describe())