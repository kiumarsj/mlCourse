import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate more complex data
np.random.seed(42)
x = np.sort(4 * np.random.rand(100) - 2)  # Random values between -2 and 2
y = 2 * x**3 - x**2 + 0.5 * x + np.sin(3 * x) + 0.5 * np.random.randn(100)

# x = np.linspace(-2,2,100)
# y = x**2

poly_features = PolynomialFeatures(degree=3)
x_reshaped = x.reshape(-1,1)
print(x)
print('--------')
print(x_reshaped)
poly_features_t = poly_features.fit_transform(x_reshaped) # transforms the features to polynominal
poly_reg = LinearRegression()
model1 = poly_reg.fit(poly_features_t, y)
y_pred = model1.predict(poly_features_t)
print(model1.coef_)

plt.scatter(x, y, label="Data")
plt.plot(x, y_pred, label="Regression Line")
plt.legend()
plt.title("Model1")
plt.show()