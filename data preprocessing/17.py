import pandas as pd # data processing
df = pd.read_csv("student_scores.csv")
x = df['Hours'].values.reshape(-1, 1)
y = df['Scores'].values.reshape(-1, 1)
print(x.dtype) # data type for each column
print(x.ndim) # number of dimensions
print(x.shape) # total number of rows and columns. it returns a typle with each index having the number of corresponding elements.

print(df.head(n=2))
print(df.tail())
print(df.describe())