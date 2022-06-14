# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# position_salaries.csv contains two feature (Non-Linear Data)
print('-------------------------------------------------')
print('Seprating features and dependent variable . . . ')
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# As dataset's rows are less, Splitting won't be necessary

# Spliting
# from sklearn.model_selection import train_test_split
# print('-------------------------------------------------')
# print('splitting . . . ')
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Displaying
print('-------------------------------------------------')
print('Length of X', len(X))
print(X)
print('-------------------------------------------------')
print('Length of y_train', len(y))
print(y)
print('-------------------------------------------------')
y = y.reshape(len(y),1)
#print('Length of y_train', len(y))
#print(y)

# Feature Scalling 
# SVR won't work without feature scalling 
# As, no explicit equation of dependent variable with respect to independent variable
print('-------------------------------------------------')
print('Scalling features . . .')
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Training
from sklearn.svm import SVR
print('-------------------------------------------------')
print('Machine is learning . . .')
regressor = SVR(kernel = 'rbf') # Recommended kernel for SVR
regressor.fit(X, y)

# Inverse Transforming and Predicting
print('-------------------------------------------------')
print('Predicting . . .')

sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(1,-1))

print('-------------------------------------------------')
print('Visualising the Result')

# Visualising the Polynomial Regression results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)) , color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()