from sklearn.preprocessing import Binarizer
x= [[10,-10,20],
    [20,10,15],
    [14,16,-19]]
bi = Binarizer(threshold=10.0).fit_transform(x)
print(bi)