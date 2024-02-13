# one hot encoding
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
df = pd.DataFrame({'color': ['red', 'green', 'blue']})
encoder = OneHotEncoder()
encoded = encoder.fit_transform(df)
print(encoded)
print(encoded.toarray())