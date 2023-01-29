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
