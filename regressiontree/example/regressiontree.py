import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the Height Weight Dataset
data = pd.read_csv('Height_Age_Dataset.csv')
print(data.head())

# Store data in dependent and independent variables
X = data.iloc[:, 0:1].values
y = data.iloc[:, 1].values

# Split the dataset into training and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Import the Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

# Create a decision tree regressor object from DecisionTreeRegressor class
DtReg = DecisionTreeRegressor(random_state=0)

# Fit the decision tree regressor with training data represented by X_train and y_train
DtReg.fit(X_train, y_train)

# Predicted Height from test dataset w.r.t DTR
y_predict_dtr = DtReg.predict(X_test)

# Model evaluation using R-Square for Decision Tree Regression
from sklearn import metrics
r_square = metrics.r2_score(y_test, y_predict_dtr)
print('R-Square Error associated with Decision Tree Regression is:', r_square)

''' Visualise the Decision Tree Regression by creating range of values from min value of X_train to max value of 
X_train having a difference of 0.01 between two consecutive values
'''
X_val = np.arange(min(X_train), max(X_train))

# Reshape the data into a len(X_val)*1 array in order to make a column out of the X_val values
X_val = X_val.reshape((len(X_val), 1))

# Define a scatter plot for training data
plt.scatter(X_train, y_train, color='blue')

# Plot predicted data
plt.plot(X_val, DtReg.predict(X_val), color='red')

# Define title
plt.title("Height prediction using DTR")

# Define X axis
plt.xlabel('Age')

# Define Y axis
plt.ylabel('Height')

plt.show()

# Import export_graphviz package
from sklearn.tree import export_graphviz

# Store the decision tree in a tree.dot file to visualize plot
# Visualize on webgraphviz.com by cping related data from dtregression.dot file
export_graphviz(DtReg, out_file='dtregression.dot', feature_names=['Age'])

# Predicting Height based on Age using Decision Tree Regression
height_pred = DtReg.predict([[41]])
print("Predicted Height: % d" % height_pred)
