from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  
import numpy as np

data = pd.read_csv("Social_Network_Ads.csv")
df = pd.DataFrame(data)
X_data = df.iloc[:, 0:-1]
y_data = df.iloc[:, -1:]

X_data = X_data.to_numpy()
y_data = y_data.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

scaler = StandardScaler()

X_train_scaler = scaler.fit_transform(X_train)
X_val_scaler = scaler.fit_transform(X_val)
X_test_scaler = scaler.fit_transform(X_test)

best_C = 0
best_gamma = 0
best_acc = 0
best_kernel = ''

for c in [0.1, 1, 10, 100, 1000]:
    for g in [1, 0.1, 0.01, 0.001, 0.0001]:
        for k in ['linear', 'rbf', 'sigmoid']:
            model = svm.SVC(C= c, gamma= g, kernel= k)
            model.fit(X_train_scaler, y_train)
            acc = model.score(X_val_scaler, y_val)
            if acc > best_acc:
                best_acc = acc
                best_C = c
                best_gamma = g
                best_kernel = k 
     

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2)

X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.fit_transform(X_test)

from sklearn.model_selection import KFold
kfold = KFold(5, shuffle = True, random_state = 1)

best_C1 = 0
best_gamma1 = 0
best_acc1 = 0
best_kernel1 = ''

for c in [0.1, 1, 10, 100, 1000]:
    for g in [1, 0.1, 0.01, 0.001, 0.0001]:
        for k in ['linear', 'rbf', 'sigmoid']:
            accuracy_list = []
            for train, val in kfold.split(X_train_scaler, y_train):
                X_train_kf, X_val_kf = X_train_scaler[train], X_train_scaler[val]
                y_train_kf, y_val_kf = y_train[train], y_train[val]

                model = svm.SVC(C= c, gamma= g, kernel= k)
                model.fit(X_train_kf, y_train_kf)

                acc = model.score(X_val_kf, y_val_kf)
                accuracy_list.append(acc)
            acc_mean = np.mean(accuracy_list)
            if acc_mean > best_acc1:
                best_acc1 = acc_mean
                best_C1 = c
                best_gamma1 = g
                best_kernel1 = k
print('\nWith train test split: Best C =', best_C, ', Best gamma = ', best_gamma, ', Best kernel = ', best_kernel)
print('\nWith KFold = 4: Best C =', best_C1, ', Best gamma = ', best_gamma1, ', Best kernel = ', best_kernel1)
