import pandas as pd

from regression_tree import *

# Import the Calgary Factors Dataset using absolute path
CALGARY_DATA = pd.read_csv(r'C:\Chung\URO\covid-factors\datasets\Calgary_Factors.csv')

r1 = RegressionTree(CALGARY_DATA)
r1.regression_tree()
r1.get_viz()
r1.get_feature_importance()
r1.get_summary()
