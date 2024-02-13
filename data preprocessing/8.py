import pandas as pd
import numpy as np
s = pd.Series([2,3,np.nan,7,"The GPA"])
print(s.isnull())
print(s.notnull().values.any())
print(s.isnull().sum())