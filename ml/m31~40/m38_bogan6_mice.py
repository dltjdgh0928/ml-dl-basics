import numpy as np
import pandas as pd
from impyute.imputation.cs import mice

# data = pd.DataFrame([[2, np.nan, 6, 8, 10],
#                     [2, 4, np.nan, 8, np.nan],
#                     [2, 4, 6, 8, 10],
#                     [np.nan, 4, np.nan, 8, np.nan]]).transpose()

data = pd.DataFrame([[2, 2, 2, np.nan], [np.nan, 4, 4, 4], [6, np.nan, 6, np.nan], [8, 8, 8, 8], [10, np.nan, 10, np.nan]])
data.columns = ['x1', 'x2', 'x3', 'x4']

impute_df = mice(data)
print(impute_df)