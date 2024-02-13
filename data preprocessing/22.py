# knn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

df = pd.read_csv('student_scoresn.csv')
x = df[['Scores']]
y = df['Pass']

plt.scatter(df['Scores'], y)
plt.xlabel('Scores')
plt.ylabel('Pass/Fail')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

knn = KNeighborsClassifier(n_neighbors=15, metric='euclidean')
knn.fit(x_train.values, y_train)

y_pred = knn.predict(x_test.values)
print(y_pred)