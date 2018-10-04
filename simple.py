#A simple neural network program for one input and output later. No hidden layer

import numpy as np

def  sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))


input = np.array([[0,1,1],[1,0,0],[1,0,1],[1,1,1]])
check = np.array([[0,1,1,1]]).T 

np.random.seed(1)

weight = 2*np.random.random((3,1))-1

for i in range(1000000):
    l0 = input 
    l1 = sigmoid(np.dot(l0,weight))

    l1_error = check-l1 
    l1_delta = l1_error * sigmoid(l1,True)
    weight = weight + np.dot(l0.T,l1_delta)

print("Output")
print(l1)
