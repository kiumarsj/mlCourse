import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# matplotlib setting
mpl.rcParams['figure.figsize'] = (4,4)
mpl.rcParams['figure.dpi'] = 150
# create sample dataset
x,y = datasets.make_classification(1000, 10, n_informative=5, class_sep=0.4)
print(x)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
clf = LogisticRegression()
_ = clf.fit(x_train, y_train)
# predict on the test set
y_pred = clf.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# plot
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
# disp.show()

# from yellowbrick.classifier import ConfusionMatrix
