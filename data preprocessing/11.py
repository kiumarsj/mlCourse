import numpy as np
from sklearn.preprocessing import MinMaxScaler

x = np.array([[1000,2],[2000,3],[3000,4],[4000,5]])
print(f'original data: \n{x}')

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler(feature_range=(-1,1))
x1 = scaler1.fit_transform(x)
x2 = scaler2.fit_transform(x)

print(f'new data: \n{x1}')
print(f'new data: \n{x2}')