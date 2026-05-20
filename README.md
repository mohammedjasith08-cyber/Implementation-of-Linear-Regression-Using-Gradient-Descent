# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and extract R&D Spend (X) and Profit (y) as input and output variables.
2. Normalize the input feature X using mean and standard deviation to improve convergence.
3.Initialize model parameters slope (m) and intercept (b) to zero. 
4. For a fixed number of epochs, compute predicted values and calculate gradients of loss w.r.t m and b.
5.Update m and b using gradient descent with a chosen learning rate to minimize error.
6.Use the trained model to predict profit for new input and visualize results with a regression line.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: MOHAMMED JASITH J
RegisterNumber:  212225230180
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Startup.csv')
X=data['R&D Spend']
y=data['Profit']

X=(X-X.mean())/( X.std())

m=0
b=0 
epochs = 1000
n=len(X)
learning_rate = 0.01
for i in range(epochs):
  y_pred = m * X + b
  dm = (-2/n)*np.sum(X *(y-y_pred))
  db = (-2/n)*np.sum(y-y_pred)
  m = m - learning_rate * dm
  b = b - learning_rate * db

y_pred = m * X + b
print(f"Slope:{m}")
print(f"Intercept:{b}")

x_input=int(input("Enter R&D Spend:"))
predicted_profit = m * x_input + b
print("Predicted Profit:", predicted_profit)

plt.scatter(X,y,label="Actual Data")
plt.plot(X,y_pred,label="Regression Line")
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.legend()
plt.title("Gradient Descent on 50 city datasets")
plt.show()
```

## Output:

<img width="1324" height="672" alt="image" src="https://github.com/user-attachments/assets/ebf205a7-a6f3-4f19-a9f4-e07b4f1c9c06" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
