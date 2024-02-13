import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.DataFrame({'color': ['red', 'green', 'blue']})
print(df)
encoder = LabelEncoder()
encoded = encoder.fit_transform(df['color'])
print(encoded)