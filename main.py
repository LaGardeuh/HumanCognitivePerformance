import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv('human_cognitive_performance.csv')

df.head()
df.describe()
df.info()

X = df.iloc[:, :-3]
y = df.iloc[:, -3:]

X = X.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')

X.fillna(0, inplace=True)
y.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

start_time = time.time()
dummy_mean = DummyRegressor(strategy='mean')
dummy_mean.fit(X_train, y_train)
end_time = time.time()

start_time = time.time()
dummy_median = DummyRegressor(strategy='median')
dummy_median.fit(X_train, y_train)
end_time = time.time()

y_pred_mean_test = dummy_mean.predict(X_test)
y_pred_median_test = dummy_median.predict(X_test)

print("Training set statistics:")
print(f"Mean of targets: {y_train.mean().values}")
print(f"Median of targets: {y_train.median().values}")

mse_mean = mean_squared_error(y_test, y_pred_mean_test)
mse_median = mean_squared_error(y_test, y_pred_median_test)
print(f'RMSE with mean strategy: {np.sqrt(mse_mean):.4f}')
print(f'RMSE with median strategy: {np.sqrt(mse_median):.4f}')


reg = LinearRegression()
reg.fit(X_train, y_train)
y_LRpred = reg.predict(X_test)

y_LRmse = mean_squared_error(y_test, y_LRpred)
print(f'RMSE with LR strategy: {np.sqrt(y_LRmse):.4f}')

rfr = RandomForestRegressor(max_depth=2, random_state=0)
rfr = rfr.fit(X_train, y_train)

y_rfrpred = rfr.predict(X_test)

y_rfrmse = mean_squared_error(y_test, y_rfrpred)
print(f'RMSE with rfr strategy: {np.sqrt(y_rfrmse):.4f}')
