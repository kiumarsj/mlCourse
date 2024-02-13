from sklearn.preprocessing import Normalizer
x= [[10,-10,20],
    [20,10,15],
    [14,16,-19]]
n = Normalizer(norm='l2')
n_scaler = n.fit_transform(x)
print(f'new data: \n{n_scaler}')