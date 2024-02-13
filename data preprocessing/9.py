import numpy as np
from sklearn.impute import SimpleImputer

data = np.array([[5,np.nan,8], [9,3,5],[8,6,4]])

print(f'original data: \n{data}')

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# strategy: mean, median, most_frequent, constant(fill_value=0)

data1=imputer.fit_transform(data)
print(f'new data: \n{data1}')