from turtle import shape
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import numpy as np

def one_hot(x):
    classes,index = np.unique(x, return_inverse = True)
    one_hot_vector = np.zeros((x.shape[0],len(classes)))
    for i, cls in enumerate (index):
        one_hot_vector[i,cls] = 1
    return one_hot_vector
'''
names =['a','b','c','a','c']
one_hot_vector = one_hot(names)
print(one_hot_vector)
'''
data = pd.read_csv('Regression dataset/Position_Salaries.csv')
print(data)
data = data.to_numpy() # chuyển đổi sang data của Numpy
X = data[:,: -1]
Y = data[:, -1]
X_onehot = one_hot(X[:,0])
print(X_onehot.shape)

transformed_X = np.concatenate([X_onehot, X[:,1:]], axis = -1)
print('tranformed_X: ', transformed_X.shape)
X_train,X_test, Y_train , Y_test = train_test_split(X, Y, test_size=0.3)
print(X_train)


model = DecisionTreeRegressor()
model2 =RandomForestRegressor()
#model = LinearRegression()
model.fit(X_train.reshape(-1,1), Y_train)
X=np.array([[1.5]]) # input WorkExperience
y = model.predict(X) 
print(y)
Y_pred = model.predict(X_test.reshape(-1,1))
l2 = ((Y_pred - Y_test)**2).sum()
l2=(l2**0.5)/Y_test.shape[0]
print("L2 loss: ", l2)

l1 = (Y_pred - Y_test)
l1 = np.absolute(l1).sum()
l1 = l1/Y_test.shape[0]
print("L1 loss: ", l1)
plt.scatter(X, Y)
plt.show()

#print(data)
