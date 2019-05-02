# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 22:43:49 2019

@author: Ebubekir
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("salary.csv",sep=";")
print(data.head())

x=data.iloc[:,0].values.reshape(-1,1)
y=data.iloc[:,1].values.reshape(-1,1)

#%% Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
tree_regression=DecisionTreeRegressor()
tree_regression.fit(x,y)

#print(tree_regression.predict(np.array([[4.7]]))) 

x2=np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head=tree_regression.predict(x2)

plt.scatter(x,y,color="red")
plt.plot(x2,y_head,color="green")
plt.xlabel("Years Experience")
plt.ylabel("Salary")

