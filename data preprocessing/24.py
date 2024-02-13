# random forest regression
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

x,y = make_regression(n_samples=10, n_features=1, noise=0.3, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42) # using 100 trees

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("mean squared error: ", mse)