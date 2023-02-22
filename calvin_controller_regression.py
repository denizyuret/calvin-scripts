# See calvin_controller_coordinates.py for an explanation.

import sys
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data from output of calvin_controller_coordinates
data = pd.read_csv(sys.stdin, delimiter='\t', header=None)

# Output some stats:
print(data.describe().to_string())

# Separate the independent and dependent variables
X = data.iloc[:, 1:2].values
x = data.iloc[:, 2].values
y = data.iloc[:, 3].values
z = data.iloc[:, 4].values

# Create a linear regression model and fit the data
xmodel = LinearRegression().fit(X, x)
ymodel = LinearRegression().fit(X, y)
zmodel = LinearRegression().fit(X, z)

# Print the coefficients of the linear regression model
print(f"x = {xmodel.coef_[0]}*input {xmodel.intercept_:+.6f}")
print(f"y = {ymodel.coef_[0]}*input {ymodel.intercept_:+.6f}")
print(f"z = {zmodel.coef_[0]}*input {zmodel.intercept_:+.6f}")
