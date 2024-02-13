# decision tree regression
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import graphviz
from math import sqrt

x,y = make_regression(n_samples=10, n_features=1, noise=0.3, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)
plt.show()

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("MSE: ", mse)
print("MAE: ", mae)
print("RMSE: ", rmse)
print("R2: ", r2)

import pydotplus
from IPython.display import Image
dot_data = export_graphviz(model, out_file=None, filled=True, rounded=True, feature_names=['Feature'])
graph = graphviz.Source(dot_data)
graph.render('23_decision_tree') # save the visualized tree to a file
graph2 = pydotplus.graph_from_dot_data(dot_data)
graph2.write_png('23.png')
Image(graph2.create_png())