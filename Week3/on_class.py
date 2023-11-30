import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
data = pd.read_csv('Regression dataset/Salary_Data.csv')
data = data.to_numpy() # chuyển đổi sang data của Numpy
X = data[:, 0]
Y = data[:, 1]

X_train,X_test, Y_train , Y_test = train_test_split(X, Y, test_size=0.3)

model = LinearRegression()
model.fit(X_train.reshape(-1,1), Y_train)
# X=np.array([[1.5]]) # input WorkExperience
# y = model.predict(X)
# print(y)
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
print(data)