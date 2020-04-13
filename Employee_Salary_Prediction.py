#!/usr/bin/env python
# coding: utf-8

# ## Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#to disable all the warnings
import warnings
warnings.filterwarnings('ignore')


# ## Importing the dataset

# In[2]:


dataset = pd.read_csv('Position_Salaries.csv')
dataset


# In[3]:


#X is a matrix
X = dataset.iloc[:, 1:-1].values
#y is a vector
y = dataset.iloc[:, -1].values


# # Training the Linear Regression model on the whole dataset

# In[4]:


#import Linear Regression class

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


# In[5]:


# Visualising the linear regression results

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# # Training the Polynomial Regression model on the whole dataset

# ### (Linear Regression model)

# In[6]:


#import Polynomial Linear Regression class

from sklearn.preprocessing import PolynomialFeatures

#polynomial degree 2

poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# In[7]:


#bias and 2 degree X matrics
X_poly


# In[8]:


# Visualising the polynomial linear regression results for polynomial 2 degreee

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[9]:


#polynomial degree 3

poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# In[10]:


#bias and 3 degree X matrics

X_poly


# In[11]:


# Visualising the polynomial linear regression results for polynomial 3 degreee

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[12]:


#polynomial degree 4

poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# In[13]:


#bias and 4 degree X matrics

X_poly


# In[14]:


# Visualising the polynomial linear regression results for polynomial 4 degreee

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[15]:


# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
# for polynomial 4 degreee

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# # Training the Support Vector Regression model on the whole dataset

# ### (Non-Linear Regression model)

# In[16]:


#X value
X


# In[17]:


#y value
y


# In[18]:


#feature Scaling for SVR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_sc = X
X_sc=sc_X.fit_transform(X_sc)

y_sc = y
y_sc=sc_y.fit_transform(y_sc.reshape(-1,1))


# In[19]:


#scaled X value
X_sc


# In[20]:


#scaled y value
y_sc


# In[21]:


#import Support Vector Regression class
#rbf kernel

from sklearn.svm import SVR
regressor_svr = SVR(kernel = 'rbf')
regressor_svr.fit(X_sc, y_sc)


# In[22]:


## Visualising the SVR results

plt.scatter(X_sc, y_sc, color = 'red')
plt.plot(X_sc, regressor_svr.predict(X_sc), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[23]:


# Visualising the SVR results (for higher resolution and smoother curve)

X_grid = np.arange(min(X_sc), max(X_sc), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_sc, y_sc, color = 'red')
plt.plot(X_grid, regressor_svr.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# # Training the Decision Tree Regression model on the whole dataset

# ### (Non-Linear & Non-Continuous Regression model)

# In[24]:


#X value
X


# In[25]:


#y value
y


# In[26]:


#import Decision Tree Regression class

from sklearn.tree import DecisionTreeRegressor
regressor_dtr = DecisionTreeRegressor(random_state = 0)
regressor_dtr.fit(X, y)


# In[27]:


# Visualising the Decision Tree Regression results (higher resolution)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor_dtr.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# # Training the Random Forest Regression model on the whole dataset

# ### (Non-Linear & Non-Continuous Regression model)

# In[28]:


#import Random Forest Regression class

from sklearn.ensemble import RandomForestRegressor

#for 10 trees
regressor_rfr1 = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor_rfr1.fit(X, y)


# In[29]:


# Visualising the Random Forest Regression results (higher resolution)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor_rfr1.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[30]:


#for 100 trees

regressor_rfr2 = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor_rfr2.fit(X, y)


# In[31]:


# Visualising the Random Forest Regression results (higher resolution)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor_rfr2.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[32]:


#for 300 trees

regressor_rfr3 = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor_rfr3.fit(X, y)


# In[33]:


# Visualising the Random Forest Regression results (higher resolution)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor_rfr3.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# ## Predicting a new result with Linear Regression

# In[34]:


y_pred=lin_reg.predict([[6.5]])
print("Salary : $",y_pred[0])


# ## Predicting a new result with Polynomial Regression

# In[35]:


#taking degree of polynomial = 4
y_pred=lin_reg_2.predict(poly_reg.fit_transform(np.array([[6.5]])))
print("Salary : $",y_pred[0])


# ## Predicting a new result with Support Vector Regression

# In[36]:


y_pred = regressor_svr.predict(sc_X.transform(np.array([[6.5]])))
y_pred = sc_y.inverse_transform(y_pred)
print("Salary : $",y_pred[0])


# ## Predicting a new result with Decision Tree Regression

# In[37]:


y_pred = regressor_dtr.predict([[6.5]])
print("Salary : $",y_pred[0])


# ## Predicting a new result with Random Forest Regression

# In[38]:


#for 10 trees

y_pred = regressor_rfr1.predict([[6.5]])
print("Salary : $",y_pred[0])


# In[39]:


#for 100 trees

y_pred = regressor_rfr2.predict([[6.5]])
print("Salary : $",y_pred[0])


# In[40]:


#for 300 trees

y_pred = regressor_rfr3.predict([[6.5]])
print("Salary : $",y_pred[0])

