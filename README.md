# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the nutrition dataset into Python using Jupyter Notebook.
2. Preprocess the data and select relevant nutritional features.
3. Train the Logistic Regression classification model.
4. Classify food items and evaluate the model performance using accuracy metrics.

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#Load the dataset
df=pd.read_csv('food_items (1).csv')

#Inspect the dataset
print('Name: Nikhil Nirmal Kumar')
print('Reg No: 212225230201')
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info)
X_raw=df.iloc[:,:-1]
y_raw=df.iloc[:,-1:]
scaler=MinMaxScaler()

#Scaling the raw input features
X=scaler.fit_transform(X_raw)

#Create a LabelEncoder object
label_encoder=LabelEncoder()

#Encode the target variable
y=label_encoder.fit_transform(y_raw.values.ravel())
#Note that ravel() functions flattens the vector.

#First, let's split the training and testing dataset
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,stratify=y,random_state=123)

#L2 penalty to shrink coefficients without removing any features from the model
penalty='l2'
#Our classification problem is multinomial
multi_class='multinomial'
#Use lbfgs for L2 penalty and multinomial classes
solver='lbfgs'
#Max iteration=1000
max_iter=1000

#Define a logistic regression model with above arguments
l2_model=LogisticRegression(random_state=123,penalty=penalty,multi_class=multi_class,solver=solver,max_iter=max_iter)
l2_model.fit(X_train,y_train)
```

## Output:
<img width="762" height="636" alt="image" src="https://github.com/user-attachments/assets/48f2359b-90ec-425d-a54e-eeff524949b1" />
<img width="953" height="823" alt="image" src="https://github.com/user-attachments/assets/24e908b4-04fd-4ea5-9559-e90da42acf1a" />
<img width="1149" height="407" alt="image" src="https://github.com/user-attachments/assets/c7037810-293e-470a-b7eb-11cd25a4e6df" />

## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
