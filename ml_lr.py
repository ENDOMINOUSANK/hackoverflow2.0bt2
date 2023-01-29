'''Geopolitics
Land degradation affects 3.2 billion people and costs the global economy around 10% of its GDP annually. Despite pledges from nations to restore 350 million hectares
of degraded land, effective execution is hindered by land disputes. A scalable solution is needed to address this issue, as more than 3 million people are impacted by
land conflicts in India alone. Using AI and socio-economic data, it is possible to identify and assess the underlying causes of these conflicts and suggest realistic 
solutions. Furthermore, it is crucial to understand how these issues can be forecasted in the future and how to reduce their economic impact.
'''
import numpy as np
import pandas as pd
import scipy as sc
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
crop = pd.read_csv(r'D:\scikit learn\data4.csv')
# print(crop.describe()) 
train, test = train_test_split(crop, test_size=.2, random_state=1)
predictors = ["CLPAST", "CLIDLE", "GL",  "FL", "MISC",  "URBAN", "OTHERS"]
target = "CLCROP"
X = train[predictors].copy()
y = train[target].copy()
lr = LinearRegression()

lr.fit(crop[["CLPAST", "CLIDLE", "GL",  "FL", "MISC",  "URBAN", "OTHERS"]], crop[["CLCROP"]])

print(lr.intercept_) 
#= [209.96203826]
print(lr.coef_)
#=[[ 1.65148294  0.32286076  0.09144891 -0.17310119 19.06623182  1.85730816 0.11783861]]

# as we r still learninig ai ml we r currently unble to plot it into a graph;)
