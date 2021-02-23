import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn import metrics

# Import the Calgary Factors Dataset
calgarydata = pd.read_csv(r'/datasets/Calgary_Factors.csv')


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
        self.DtReg = DecisionTreeRegressor(max_depth=10, min_samples_split=5, min_samples_leaf=1, random_state=0)
        self.DtReg.fit(self.x_train, self.y_train)
        y_predict_dtr = self.DtReg.predict(self.x_test)
        r_square = metrics.r2_score(self.y_test, y_predict_dtr)
        print(r_square)

    def get_feature_importance(self):
        # get importance
        importance = self.DtReg.feature_importances_

        # summarize feature importance
        for i, v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))

        # plot feature importance
        plt.bar([x for x in range(len(importance))], importance)
        plt.show()

    def get_viz(self):
        # Store the decision tree in a tree.dot file to visualize plot
        # Visualize on webgraphviz.com by cping related data from dtregression.dot file

        export_graphviz(self.DtReg, out_file='calgarydtregression.dot', feature_names=["Mean Temp (C)",
                        "Total Precip (mm)", "Avg Rel Hum (%)", "Avg Wind Spd (km/h)", "Daylight (hrs)", "Mean UV"])

    # def getDtReg(self):
    #     print(self.DtReg)
    #
    # def getx_train(self):
    #     print(self.x_train)

    # def regression_tree_plt(self):
    #     X_val = np.arange(min(self.x_train), max(self.x_train))
    #
    #     # Reshape the data into a len(X_val)*1 array in order to make a column out of the X_val values
    #     X_val = X_val.reshape((len(X_val), 1))
    #
    #     # Define a scatter plot for training data
    #     plt.scatter(self.x_train, self.y_train, color='blue')
    #
    #     # Plot predicted data
    #     plt.plot(X_val, self.DtReg.predict(X_val), color='red')
    #
    #     # Define title
    #     plt.title("COVID Cases prediction using DTR")
    #
    #     # Define Y axis
    #     plt.ylabel('Number of Covid Cases')
    #
    #     plt.show()


r1 = RegressionTree(calgarydata)
r1.regression_tree()
# r1.get_viz()
r1.get_feature_importance()
