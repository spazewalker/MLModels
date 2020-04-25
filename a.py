import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
alpha = 0.1

data = pd.read_csv('/home/shivanshu/Desktop/ML_implementation/insurance.csv')
data = data.replace('female',1)
data = data.replace('male',0)
data = data.replace('yes',1)
data = data.replace('no',0)
data = data.drop("region",axis=1)
(m,n) = data.shape
test = data[int(m*0.8):m].to_numpy()
test = np.transpose(test)
train = data[:int(m*0.8)].to_numpy()
train = np.transpose(train)
Y = train[-1:,:]
X = train[:-1,:]
(n,m) = X.shape
temp = np.ones(m)
temp = temp.reshape(1,m)
X = np.append(temp,X,axis=0)

def activate(theta,X):
    return np.transpose(theta)@X

def predict(X,theta):
    A = X
    for theta_t in theta:
        # print(A.shape,theta_t.shape)
        A = activate(theta_t,A)
    return A

def calculateCost(X,theta,Y):
    (n,m) = X.shape
    return sum(sum((predict(X,theta)-Y)**2))/2/m

theta1 = np.random.randn(6,6)
theta2 = np.random.randn(6,1)
print(X.shape)
predict(X,(theta1,theta2))
cost = calculateCost(X,(theta1,theta2),Y)
print(cost)

costs = []
costs.append(cost)
for i in range(1):
    (n,m) = X.shape
    delta2 = np.transpose((predict(X,(theta1,theta2))-Y)@np.transpose(np.transpose(theta1)@X)/m)
    print(delta2.shape,delta2)
    delta1 = (predict(X,(theta1,theta2))-Y)
    print(delta1.shape,delta1)
    print("hellp")