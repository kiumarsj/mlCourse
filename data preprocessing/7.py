import pandas as pd
df = pd.read_csv('student_scores.csv')

b = [1,10,20,30,40, 50, 60, 70, 80, 90, 100]
l = ['bin1','bin2','bin3','bin4', 'bin5', 'bin6', 'bin7', 'bin8', 'bin9', 'bin10']
df1 = pd.cut(df['Scores'],bins=b,labels=l)
print(df1)