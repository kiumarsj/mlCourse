import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder

OHencoder = OneHotEncoder()
data = np.array([[100,1],[49,0], [75,1], [80,1], [66,1],[90,1],[35,0],[60,1],[88,1],[42,0]])
df = pd.DataFrame(data, columns=['marks','passed'])


OHencoded = OHencoder.fit_transform(df)
print("One Hot Encoding")
print(OHencoded)
bi = Binarizer(threshold=10.0).fit_transform(data)
print("Binarizer")
print(bi)