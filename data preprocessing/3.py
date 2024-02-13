from sklearn.preprocessing import LabelEncoder
x=['bmw', 'mercedes', 'bmw', 'audi']
enc=LabelEncoder()
y=enc.fit_transform(x)
print(y)