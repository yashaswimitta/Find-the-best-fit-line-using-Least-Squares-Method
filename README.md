# Implementation of Univariate Linear Regression
## AIM:
To implement univariate Linear Regression to fit a straight line using least squares.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y.
2. Calculate the mean of the X -values and the mean of the Y -values.
3. Find the slope m of the line of best fit using the formula. 
<img width="231" alt="image" src="https://user-images.githubusercontent.com/93026020/192078527-b3b5ee3e-992f-46c4-865b-3b7ce4ac54ad.png">
4. Compute the y -intercept of the line by using the formula:
<img width="148" alt="image" src="https://user-images.githubusercontent.com/93026020/192078545-79d70b90-7e9d-4b85-9f8b-9d7548a4c5a4.png">
5. Use the slope m and the y -intercept to form the equation of the line.
6. Obtain the straight line equation Y=mX+b and plot the scatterplot.

## Program:
```
/*
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: B DHIVYA SHRI
RegisterNumber:  212221230009
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("ex1.txt",header=None)
data

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit($10,000)")
plt.title("Profit prediction")

def computeCost(X,y,theta):
  m = len(y)
  h = X.dot(theta)
  square_err = (h-y)**2
  return 1/(2*m) * np.sum(square_err)

  data1 = data.values
m = data1[:,0].size
X = np.append(np.ones((m,1)),data1[:,0].reshape(m,1),axis=1)
y = data1[:,1].reshape(m,1)
theta = np.zeros((2,1)) 
print("The value is : ",computeCost(X,y,theta))


def gradientdescent(X,y,theta,alpha,numiter):
  m = len(y)
  hist = []

  for i in range(numiter):
    predictions = X.dot(theta)
    error = np.dot(X.transpose(),(predictions-y))
    descent = alpha * 1/m * error
    theta -= descent
    hist.append(computeCost(X,y,theta)) 
  return theta,hist

  theta ,hist = gradientdescent(X,y,theta,0.01,1500)
print("h(x) = "+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.figure(figsize = (7,7))
plt.plot(hist)
plt.xlabel("Iterations")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.figure(figsize = (7,7))
plt.scatter(data[0],data[1])
xval = [x for x in range(25)]
yval = [y*theta[1] + theta[0] for y in xval]
plt.plot(xval,yval,color='purple')
plt.xticks(np.arange(5.30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit ($ 10,1000)")
plt.title("Profit Prediction")

def predict(x,theta):
  predictions = np.dot(theta.transpose(),x)
  return predictions[0]

predict1 = predict(np.array([1,3.5]),theta)*10000
print("For population of 35,000, we pewdict a profit of $ "+str(round(predict1,0)))

predict2 = predict(np.array([1,7]),theta)*10000
print("For population of 70,000, we pewdict a profit of $ "+str(round(predict2,0)))

```

## Output:
<img width="940" alt="ML" src="https://user-images.githubusercontent.com/94505585/194759855-c38ae9ac-3ce5-4024-9619-5fddd21d240c.png">
<img width="893" alt="2022-10-09 (6)" src="https://user-images.githubusercontent.com/94505585/194759883-be2dd70b-b90f-4197-a079-37a33e12dc7d.png">
<img width="932" alt="2022-10-09 (7)" src="https://user-images.githubusercontent.com/94505585/194759906-2a6fb81c-adda-457a-bd9e-99a7590063a3.png">
<img width="934" alt="2022-10-09 (8)" src="https://user-images.githubusercontent.com/94505585/194759920-15d3f779-368f-46c0-ad8a-07bed814e831.png">
<img width="777" alt="2022-10-09 (9)" src="https://user-images.githubusercontent.com/94505585/194759930-064cefcf-262f-47a9-abc4-3ec317b5c99d.png">


## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
