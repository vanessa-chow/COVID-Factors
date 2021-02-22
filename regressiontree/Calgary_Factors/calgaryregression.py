import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn import metrics

# Import the Calgary Factors Dataset
calgarydata = pd.read_csv(r'C:\Chung\URO\covid-factors\datasets\Calgary_Factors.csv')


class RegressionTree:

    def __init__(self, data):
        self.data = data
        self.x = data[["Mean Temp (C)", "Total Precip (mm)", "Avg Rel Hum (%)", "Avg Wind Spd (km/h)", "Daylight (hrs)",
                       "Mean UV"]]
        self.y = data[["Cases"]]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3,
                                                                                random_state=0)
        self.DtReg = None

    def regression_tree(self):
        self.DtReg = DecisionTreeRegressor(min_samples_split=10, min_samples_leaf=10, random_state=0)
        self.DtReg.fit(self.x_train, self.y_train)
        y_predict_dtr = self.DtReg.predict(self.x_test)
        r_square = metrics.r2_score(self.y_test, y_predict_dtr)
        print(r_square)

    def regression_tree_plt(self):
        X_val = np.arange(min(self.x_train), max(self.x_train))

        # Reshape the data into a len(X_val)*1 array in order to make a column out of the X_val values
        X_val = X_val.reshape((len(X_val), 1))

        # Define a scatter plot for training data
        plt.scatter(self.x_train, self.y_train, color='blue')

        # Plot predicted data
        plt.plot(X_val, self.DtReg.predict(X_val), color='red')

        # Define title
        plt.title("COVID Cases prediction using DTR")

        # Define Y axis
        plt.ylabel('Number of Covid Cases')

        plt.show()


r1 = RegressionTree(calgarydata)
r1.regression_tree()
r1.regression_tree_plt()

#
# def regression_tree(data):
#     x = data[["Mean Temp (C)", "Total Precip (mm)", "Avg Rel Hum (%)", "Avg Wind Spd (km/h)", "Daylight (hrs)", "Mean UV"]]
#     y = data[["Cases"]]
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
#     DtReg = DecisionTreeRegressor(min_samples_split=10, min_samples_leaf=10, random_state=0)
#     DtReg.fit(x_train, y_train)
#     y_predict_dtr = DtReg.predict(x_test)
#     r_square = metrics.r2_score(y_test, y_predict_dtr)
#     print(r_square)


#
# ''' Visualise the Decision Tree Regression by creating range of values from min value of X_train to max value of
# X_train having a difference of 0.01 between two consecutive values
# '''
# X_val = np.arange(min(X_train), max(X_train))
#
# # Reshape the data into a len(X_val)*1 array in order to make a column out of the X_val values
# X_val = X_val.reshape((len(X_val), 1))
#
# # Define a scatter plot for training data
# plt.scatter(X_train, y_train, color='blue')
#
# # Plot predicted data
# plt.plot(X_val, DtReg.predict(X_val), color='red')
#
# # Define title
# plt.title("COVID Cases prediction using DTR based on Temperature")
#
# # Define X axis
# plt.xlabel('Temperature')
#
# # Define Y axis
# plt.ylabel('Number of Covid Cases')
#
# plt.show()
#
#
#
