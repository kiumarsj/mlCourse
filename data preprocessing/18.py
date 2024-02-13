import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # to apply the linear regression
from sklearn.model_selection import train_test_split # to split the data into two parts
df = pd.read_csv("student_scores.csv") # load data set
x = df['Hours'].values.reshape(-1, 1) # -1 means that calculate the dimension of rows, but have 1 column
y = df['Scores'].values.reshape(-1, 1) # -1 means that calculate the dimension of rows, but have 1 column
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10) # 30% of the data is used for testing
lr = LinearRegression() # create a linear regression object
# train the model
lr.fit(x_train, y_train) # fit the model
x_test1 = [[5.1]]
y_pred1 = lr.predict(x_test1)
print(y_pred1)
y_pred = lr.predict(x_test) # make predictions
print(y_pred)
print("value of the intercept:", lr.intercept_)
print("value of the coeffient:", lr.coef_)
reg_line = f'y = {lr.intercept_} + {lr.coef_}'
print(reg_line)
