import numpy as np
import pandas as pd
from datetime import datetime
import datetime

dates = ['4/25/2023', '4/26/2023', '4/27/2023', '4/28/2023', '4/29/2023', '4/30/2023']
dates = pd.to_datetime(dates)
print(dates)
print(type(dates))

ts = pd.Series([2, None, np.nan, 8, 10, np.nan], index=dates)
ts = ts.interpolate()
print(ts)
