import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Data import
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

#print('t: ', target.shape) 
#print('raw: ', raw_df.shape)
#print('data: ', data.shape)


# Training Validation Splitting
x = data
y = target

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=22)

#print(xtrain, xtest, ytrain, ytest)


# Model Training
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(xtrain, ytrain)

y_pred = lr.predict(xtest)


# Model Accuracy
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(ytest, y_pred)

print(mse) ## Accuracy = 79 %
