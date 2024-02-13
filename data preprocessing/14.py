from sklearn.preprocessing import RobustScaler
x= [[10,-10,20],
    [20,10,15],
    [14,16,-19]]
rs = RobustScaler()
rs_scaler = rs.fit_transform(x)
print(rs_scaler)