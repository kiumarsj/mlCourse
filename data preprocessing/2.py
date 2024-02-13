from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder(categories=[['first', 'second', 'third', 'forth']])
x = [['third'], ['second'], ['first']]
y = enc.fit_transform(x)
print(y)