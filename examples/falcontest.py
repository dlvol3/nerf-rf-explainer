# An Falcon+NERF test
# Using In silico toy dataset from Falcon
# NERF V0.3
# Apr. 2020
# Yue Zhang <yue.zhang@lih.lu>

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
import platform
import time
import mygene
from sklearn.utils.multiclass import unique_labels
import os
import networkx as nx
import pickle
from scipy import stats
import re
# %%
if platform.system() == 'Windows':
    # Windows in the lab
    # gdscic = pd.read_csv('P:/VM/Drug/data/output/GDSCIC50.csv')
    falcondata_raw = pd.read_csv('C:/Users/Yue/Documents/MATLAB/falconinputf5.csv')
#%% 1 event processing
falcondata_raw['event'].quantile(0.3)
falcondata_raw['event'].quantile(0.7)

falcondata_raw['event'].plot.hist()
plt.show()
falcondata_raw['class'] = 'neutral'
falcondata_raw['class'].loc[falcondata_raw['event'] < falcondata_raw['event'].quantile(0.3)] = 'positive'
falcondata_raw['class'].loc[falcondata_raw['event'] > falcondata_raw['event'].quantile(0.7)] = 'negative'

falconready = falcondata_raw.loc[falcondata_raw['class'] != 'neutral']
falcontogo = falconready.drop('event', axis = 1)

#%%
# 2 event dataset preprocessing

falcondata_raw['event1'].quantile(0.3)
falcondata_raw['event1'].quantile(0.7)

falcondata_raw['event1'].plot.hist()

falcondata_raw['event2'].quantile(0.3)
falcondata_raw['event2'].quantile(0.7)

falcondata_raw['event2'].plot.hist()
plt.show()
falcondata_raw['class'] = 'neutral'
falcondata_raw['class'].loc[(falcondata_raw['event1'] < falcondata_raw['event1'].quantile(0.3)) & (falcondata_raw['event2'] < falcondata_raw['event2'].quantile(0.3))] = 'positive'
falcondata_raw['class'].loc[(falcondata_raw['event1'] > falcondata_raw['event1'].quantile(0.7)) & (falcondata_raw['event2'] > falcondata_raw['event2'].quantile(0.7))] = 'negative'

falconready = falcondata_raw.loc[falcondata_raw['class'] != 'neutral']
falcontogo = falconready.drop(['event1','event2'], axis = 1)

# %%
# -------------------------###
# Get familiar with python DS
# ref: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

# Label the class
le = LabelEncoder()
le_count = 0

# iterate through columns
for col in falcontogo:
    if falcontogo.loc[:, col].dtype == 'object':
        # if less than 2 classes(which is better to use one-hot coding if not)
        if len(list(falcontogo.loc[:, col].unique())) <= 2:
            # 'train' the label encoder with the training data
            le.fit(falcontogo.loc[:, col])
            # Transform both training and testing
            falcontogo.loc[:, col] = le.transform(falcontogo.loc[:, col])
            # pdC.loc[:, col] = le.transform(pdC.loc[:, col])

            # Keep track of how many columns were labeled
            le_count += 1

print('%d columns were label encoded.' % le_count)

# %%
# Exploratory Data Analysis(EDA)

# Distribution of the target classes(columns)
falcontogo['class'].value_counts()
falcontogo['class'].head(4)

falcontogo['class'].plot.hist()
plt.show()


# %%
# Examine Missing values
def missing_value_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing values', 1: '% of Total Values'}
    )

    # Sort the table by percentage of the missing values
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print the summary
    print("Your selected data frame has " + str(df.shape[1]) + " columns.\n"
                                                               "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the result
    return mis_val_table_ren_columns


# Check the missing value in the dataset

Missing_values = missing_value_table(falcontogo)
Missing_values.head(10)
# %%
# Column Types
# Number of each type of column
falcontogo.dtypes.value_counts()

# Check the number of the unique classes in each object column
falcontogo.select_dtypes('object').apply(pd.Series.nunique, axis=0)

# %%
# Correlations
correlations = falcontogo.corr()['class'].sort_values(na_position='first')

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))
# Create Cross-validation and training/testing


# %%
# Random forest 1st

# Define the RF
random_forest = RandomForestClassifier(n_estimators=200, random_state=123, max_features="sqrt",
                                       criterion="gini", oob_score=True, n_jobs=10, max_depth=9,
                                       verbose=0)
# %%
# Drop SENRES

train_labels = falcontogo.loc[:, "class"]


if 'class' in falcontogo.columns:
    train = falcontogo.drop(['class'], axis=1)
else:
    train = falcontogo.copy()
# train.iloc[0:3,0:3]
features = list(train.columns)
# train["SENRES"] = train_labels


# %%

# RF 1st train 5 trees

random_forest.fit(train, train_labels)

# Extract feature importances
feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})
#
# feature_importances
# train.shape
# # Make predictions on the test data
# test_labels = lapaG.loc[:, "SENRES"]
# cell_lines_lapaG = lapaG.loc[:, "gdsc.name"]
#
# if 'SENRES' in lapaG.columns:
#     test = lapaG.drop(['SENRES'], axis=1)
# else:
#     test = lapaG.copy()
#
# test = test.drop(['gdsc.name'], axis=1)
# predictions = random_forest.predict(test)
# predictions
#
# confusion_matrix(test_labels, predictions)
# %%
random_forest.oob_score_

idxs = [129]

testC = train.iloc[idxs, :]

colon_original = train.reset_index(drop = True)
colon_original_label = train_labels.reset_index(drop = True)[idxs].values
testP = random_forest.predict(train)
# testP = random_forest.predict(testC)

cfm = confusion_matrix(train_labels,testP)

cfm

featurelist = train.columns.values.tolist()
index = list(range(len(featurelist)))

# %%
# RUN NERF
pd_ff = flatforest(random_forest, testC)
pd_f = extarget(random_forest, testC, pd_ff)
pd_nt = nerftab(pd_f)

g_localnerf = localnerf(pd_nt, 0)
g_twonets = twonets(g_localnerf, str('f5-212'), index, featurelist, index1=2, index2=3)

g_pagerank = g_localnerf.replace(index, featurelist)
xg_pagerank = nx.from_pandas_edgelist(g_pagerank, "feature_i", "feature_j", "EI")
DG = xg_pagerank.to_directed()
rktest = nx.pagerank(DG, weight='EI')
rkdata = pd.Series(rktest, name='position')
rkdata.index.name = 'PR'
# rkdata
rkrank = sorted(rktest, key=rktest.get, reverse=True)
fea_corr = rkrank[0:20] # change to top 100 to get more obj. result
# top50[sampleindex] = fea_corr
# rkrank = [featurelist[i] for i in rkrank]
fn = "pagerank_sample_" + str('f5-212') + ".txt"
with open(os.getcwd() + '/output/falcon/' + fn, 'w') as f:
    for item in rkrank:
        f.write("%s\n" % item)

# %%
# Feature importance list

feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': featurelist, 'importance': feature_importance_values})
feature_importances.to_csv(os.getcwd() + '/output/falcon/falcon_f5.txt', sep='\t')


# %%

import matplotlib.pyplot as plt

correlations = falcontogo.corr()
plt.matshow(falcontogo.iloc[:, 1:len(falcontogo.columns)].corr())
plt.show()

sol = (
    correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))



