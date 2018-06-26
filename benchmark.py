#%%
from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import statsmodels.api as sm

#%%
train_features = pd.read_csv('dengue_features_train.csv', index_col=[0,1,2])
train_labels = pd.read_csv('dengue_labels_train.csv', index_col=[0,1,2])
a = train_features.loc[1991]
