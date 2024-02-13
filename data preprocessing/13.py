from sklearn.preprocessing import StandardScaler
x= [[10,-10,20],
    [20,10,15],
    [14,16,-19]]
ss = StandardScaler()
ss_scaler = ss.fit_transform(x)
print(ss_scaler)