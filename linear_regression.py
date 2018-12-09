#importing all the packages
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

#x axis value
#linear regression model accepts x axis value only in 2D array
X = [[5],[6],[4],[6],[5],[6],[7]]
Y = [55,65,40,53,58,60,70]
plt.scatter(X,Y,color='black')

#regression model
#reg.coef_ give us the slope
#reg.intercept_ give us the constant value in y=mx+c
reg = linear_model.LinearRegression()
reg.fit(X,Y)
m = reg.coef_
print(m)
c = reg.intercept_
print(c)

#plotting the regression line
x_predicted = X
y_predicted = [m*i + c for i in X]
plt.plot(x_predicted,y_predicted, color='blue')

#to predict a y value for a given x value we use
print(reg.predict(6.6))

#plotting all the graphs
plt.show()