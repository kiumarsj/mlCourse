import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

data = np.array(['low', 'medium','high'])
print(data)
encoder = OrdinalEncoder()
result = encoder.fit_transform(data.reshape(-1, 1))
print(result)
print('------------------')
x2 = pd.DataFrame({'animals':['low', 'med', 'low', 'high', 'low', 'high']})
print(x2)
enc = OrdinalEncoder()
y2 = enc.fit_transform(x2)
print(y2)
