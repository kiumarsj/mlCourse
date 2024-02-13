import pandas  as pd
import numpy   as np
s = pd.Series([2,3,np.nan,7,"The Hobbit"])
s1=s.fillna(5)
s2=s.fillna(method='ffill')
s3=s.fillna(method='bfill')

print(f'original data: \n{s}')
print(f'new data: \n{s1}')
print(f'new data: \n{s2}')
print(f'new data: \n{s3}')