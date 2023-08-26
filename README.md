# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

# AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

# Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.
# Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Akshayaa M
RegisterNumber: 212222230009 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
#displaying predicted values
Y_pred
#displaying actual values
Y_test
#graph plot for training data
plt.scatter(X_train,Y_train,color="blue")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours VS Scores (Traning Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#graph plot for test data
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours VS Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

# Output:
### df.head()
![implementation-of-simple-linear-regression-model-for-predicting-the-marks-scored](1.png)
### df.tail()
![implementation-of-simple-linear-regression-model-for-predicting-the-marks-scored](2.png)
### Array values of X
![implementation-of-simple-linear-regression-model-for-predicting-the-marks-scored](3.png)
### Array values of Y
![implementation-of-simple-linear-regression-model-for-predicting-the-marks-scored](4.png)
### Array values of Y prediction
![implementation-of-simple-linear-regression-model-for-predicting-the-marks-scored](5.png)
### Array values of Y test
![implementation-of-simple-linear-regression-model-for-predicting-the-marks-scored](6.png)
### Training set graph
![implementation-of-simple-linear-regression-model-for-predicting-the-marks-scored](7.png)
### Test set graph
![implementation-of-simple-linear-regression-model-for-predicting-the-marks-scored](8.png)
### Values for MSE,MAE,RMSE
![implementation-of-simple-linear-regression-model-for-predicting-the-marks-scored](9.png)
# Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
