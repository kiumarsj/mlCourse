import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import sessionmaker

DATABASE_FILE = "uniproject.db"
if os.path.exists(DATABASE_FILE):
    os.remove(DATABASE_FILE)

# Pydantic model for data manipulation
class StudentData(BaseModel):
    presence: int
    class_activity: int
    study_hours: int
    midterm_score: int
    project_score: int
    status: str

# Pydantic model for database schema
class Student(BaseModel):
    id: int
    presence: int
    class_activity: int
    study_hours: int
    midterm_score: int
    project_score: int
    status: str

# SQLite database setup
DATABASE_URL = "sqlite:///uniproject.db"
engine = create_engine(DATABASE_URL)

class Base(DeclarativeBase):
    pass


# Define the Student table
class StudentTable(Base):
    __tablename__ = "student"
    id = Column(Integer, primary_key=True, index=True)
    presence = Column(Integer)
    class_activity = Column(Integer)
    study_hours = Column(Integer)
    midterm_score = Column(Integer)
    project_score = Column(Integer)
    status = Column(String)

# Create the table
Base.metadata.create_all(bind=engine)

# Insert data into the database
Session = sessionmaker(bind=engine)
session = Session()

data = {
    'presence': [1, 1, 0, 0, 2, 1, 0, 4, 1, 2, 0, 0, 0, 0, 1, 2, 1, 3, 3, 2, 4, 2, 5, 0],
    'class_activity': [50, 60, 20, 90, 95, 15, 11, 27, 40, 60, 80, 70, 65, 86, 90, 45, 37, 27, 30, 60, 75, 84, 40, 66],
    'study_hours': [30, 15, 2, 40, 42, 28, 8, 35, 26, 23, 20, 38, 18, 7, 35, 29, 37, 27, 15, 30, 34, 43, 23, 5],
    'midterm_score': [10, 20, 8, 30, 38, 2, 5, 11, 10, 20, 35, 32, 23, 14, 7, 8, 5, 37, 8, 18, 38, 2, 12, 27],
    'project_score': [25, 30, 30, 20, 0, 10, 20, 30, 10, 15, 20, 15, 15, 0, 25, 15, 30, 30, 15, 0, 10, 30, 15, 30],
    'status': ['pass', 'pass', 'fail', 'pass', 'pass', 'fail', 'fail', 'fail', 'fail', 'fail', 'pass', 'pass', 'fail',
               'fail', 'fail', 'fail', 'pass', 'pass', 'fail', 'pass', 'fail', 'fail', 'fail', 'pass']
}

for i in range(len(data['presence'])):
    student_data = {field: data[field][i] for field in StudentTable.__table__.columns.keys() if field != 'id'}
    student = StudentTable(**student_data)
    session.add(student)

session.commit()
session.close()

# Read data from the database
dataframe = pd.read_sql_table("student", engine)
# dataframe = pd.read_sql_table("student", engine, columns=StudentTable.__table__.columns.keys()[1:])

# Show dataframe from the database
print(dataframe)

# Predict for new data using Pydantic model
kj_data = {
    'presence': 1,
    'class_activity': 85,
    'study_hours': 26,
    'midterm_score': 29,
    'project_score': 29,
    'status': ''
}
thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Train the model
encoder = LabelEncoder()
X = dataframe[['presence', 'class_activity', 'study_hours', 'midterm_score', 'project_score']]
y = encoder.fit_transform(dataframe['status'])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

model = LogisticRegression()
model.fit(x_train, y_train)

# y_prob = model.predict_proba(x_test)
# for threshold in thresholds:
#     y_pred = (y_prob[:, 1] > threshold).astype(int)
#     print(f"predict with the threshold {threshold}: \n {y_pred} \ntest: \n {y_test} \n")

kj = StudentData(**kj_data)
df = pd.DataFrame([kj.model_dump()])
y_prob_new = model.predict_proba(df[X.columns])
th_new = 0.7
y_pred_new = (y_prob_new[:, 1] > th_new).astype(int)

if y_pred_new[0] == 1:
    print(f'will pass this course')
else:
    print('there is no else.')