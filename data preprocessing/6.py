from numpy import asarray
from sklearn.preprocessing import OneHotEncoder
data = asarray([['red'], ['green'], ['blue']])
print(data)
encoder = OneHotEncoder(drop='first', sparse_output=False)
onehot = encoder.fit_transform(data)
print(onehot)