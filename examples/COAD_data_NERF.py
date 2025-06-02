######


#%%
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
#%%

# import all the data

# RNA COAD

coadrna_raw = pd.read_table("L:\projectphd\TCGA\HiSeqV2_TCGA_COAD")
coadrna_t = coadrna_raw.T

# reassign column and row names

coadrna_t.columns = coadrna_t.iloc[0,:]

coadrna_t["sampleID"] =coadrna_t.index   # will be removed later?> store in sample names

coadrna_t = coadrna_t.iloc[1::,::]

# COAD phenotypes

coadpheno = pd.read_table("L:\projectphd\TCGA\COAD_clinicalMatrix")

coadhisto = coadpheno.loc[:, ["histological_type", "sampleID"]]
coadhisto = coadhisto.loc[(coadhisto["histological_type"].notnull(),)]
coadhisto = coadhisto.loc[(coadhisto["histological_type"] != '[Discrepancy]',)]

#%% merge these two into one file

coad = coadhisto.merge(coadrna_t, on='sampleID', how='left')
# coad.shape

coad = coad.dropna()
coad = coad.reset_index()
coad_train = coad.drop(["index","sampleID"],axis=1)

#%% model training

# -------------------------###
# Get familiar with python DS
# ref: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

# Label the class
# le = LabelEncoder()
# le_count = 0
#
# # iterate through columns
# for col in coad_train:
#     if coad_train.loc[:, col].dtype == 'object':
#         # if less than 2 classes(which is better to use one-hot coding if not)
#         if len(list(coad_train.loc[:, col].unique())) <= 2:
#             # 'train' the label encoder with the training data
#             le.fit(coad_train.loc[:, col])
#             # Transform both training and testing
#             coad_train.loc[:, col] = le.transform(coad_train.loc[:, col])
#             # pdC.loc[:, col] = le.transform(pdC.loc[:, col])
#
#             # Keep track of how many columns were labeled
#             le_count += 1
#
# print('%d columns were label encoded.' % le_count)


#%%  as many columns also have less than 3 unique values we have to assign hot code by hand
le = LabelEncoder()
le.fit(coad_train.loc[:, 'histological_type'])
coad_train.loc[:, 'histological_type'] = le.transform(coad_train.loc[:, 'histological_type'])

# %%
# Exploratory Data Analysis(EDA)

# Distribution of the target classes(columns)
coad_train['histological_type'].value_counts()
coad_train['histological_type'].head(4)

coad_train['histological_type'].plot.hist()
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

Missing_values = missing_value_table(coad_train)
Missing_values.head(10)
# %%
# Column Types
# Number of each type of column
coad_train.dtypes.value_counts()

# Check the number of the unique classes in each object column
coad_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)

# %%
# Correlations
# correlations = coad_train.corr()['histological_type']
#
# # Display correlations
# print('Most Positive Correlations:\n', correlations.tail(15))
# print('\nMost Negative Correlations:\n', correlations.head(15))

# Create Cross-validation and training/testing


# %%
# Random forest 1st

# Define the RF
random_forest = RandomForestClassifier(n_estimators=300, random_state=123, max_features="sqrt",
                                       criterion="gini", oob_score=True, n_jobs=10, max_depth=10,
                                       verbose=0)
# %%
# Drop SENRES

train_labels = coad_train.loc[:, "histological_type"]


if 'histological_type' in coad_train.columns:
    train = coad_train.drop(['histological_type'], axis=1)
else:
    train = coad_train.copy()
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

# #############################################################################
# Classification and ROC analysis
#%%

# #############################################################################
# Data IO and generation

# Import some data to play with
# X = train
# y = train_labels
# X = X.values
# y = y.values
# #
# # n_samples, n_features = X.shape
# # X = x_test
# # y = y_test
# # X = X.values
# # y = y.values
# from scipy import interp
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import StratifiedKFold
# # Run classifier with cross-validation and plot ROC curves
# cv = StratifiedKFold(n_splits=6)
# classifier = random_forest
#
# tprs = []
# aucs = []
# mean_fpr = np.linspace(0, 1, 100)
#
# i = 0
# for train, test in cv.split(X, y):
#     probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
#     # Compute ROC curve and area the curve
#     fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
#     tprs.append(interp(mean_fpr, fpr, tpr))
#     tprs[-1][0] = 0.0
#     roc_auc = auc(fpr, tpr)
#     aucs.append(roc_auc)
#     plt.plot(fpr, tpr, lw=1, alpha=0.3,
#              label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
#
#     i += 1
# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#          label='Chance', alpha=.8)
#
# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# std_auc = np.std(aucs)
# plt.plot(mean_fpr, mean_tpr, color='b',
#          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#          lw=2, alpha=.8)
#
# std_tpr = np.std(tprs, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                  label=r'$\pm$ 1 std. dev.')
#
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic CV')
# plt.legend(loc="lower right")
# plt.savefig('Per.png')
# plt.show()

#%%
#idxs = [3:46]
idxs = list(range(3,46,1))

testC = train.iloc[idxs, :]

colon_original = train.reset_index(drop = True)
colon_original_label = train_labels.reset_index(drop = True)[idxs].values
# testP = random_forest.predict(train)
testP = random_forest.predict(testC)

#cfm = confusion_matrix(train_labels[idxs],testP)

cfm

featurelist = train.columns.values.tolist()
index = list(range(len(featurelist)))

# %%
# RUN NERF
pd_ff = flatforest(random_forest, testC)
pd_f = extarget(random_forest, testC, pd_ff)
pd_nt = nerftab(pd_f)


##### branch to line 334
g_localnerf = localnerf(pd_nt, 2)
g_twonets = twonets(g_localnerf, str('no68_c0'), index, featurelist, index1=6, index2=6)

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
fn = "pagerank_sample_" + str('no68_c0') + ".txt"
with open(os.getcwd() + '/output/falcon/' + fn, 'w') as f:
    for item in rkrank:
        f.write("%s\n" % item)

# %%
# Feature importance list

feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': featurelist, 'importance': feature_importance_values})
feature_importances.to_csv(os.getcwd() + '/output/falcon/coad.txt', sep='\t')

#%%

from sklearn import tree
tree.plot_tree(random_forest.best_estimator_.estimators_[k])


#%%
# %% all colons

top50 = pd.DataFrame()

for sampleindex in range(testC.shape[0]):
    g_localnerf = localnerf(pd_nt, sampleindex)
    g_twonets = twonets(g_localnerf, str(sampleindex), index, featurelist, index1=5, index2=6)
    # pagerank
    g_pagerank = g_localnerf.replace(index, featurelist)
    xg_pagerank = nx.from_pandas_edgelist(g_pagerank, "feature_i", "feature_j", "EI")
    DG = xg_pagerank.to_directed()
    rktest = nx.pagerank(DG, weight='EI')
    rkdata = pd.Series(rktest, name='position')
    rkdata.index.name = 'PR'
    rkdata
    rkrank = sorted(rktest, key=rktest.get, reverse=True)
    fea_corr = rkrank[0:50] # change to top 100 to get more obj. result?
    top50[sampleindex] = fea_corr
    # rkrank = [featurelist[i] for i in rkrank]
    fn = "pagerank_sample_" + str(sampleindex) + ".txt"
    with open(os.getcwd() + '/output/falcon/' + fn, 'w') as f:
        for item in rkrank:
            f.write("%s\n" % item)



#%%
# Pairwise RBO
# remove duplicates

# top50 = top50.iloc[:, np.r_[0:16,17,18,21]]

 rbo(top50.iloc[:, 0], top50.iloc[:, 1], p=0.9)['ext']
 corr_rbo_matrix = pd.DataFrame(np.zeros((top50.shape[1], top50.shape[1])))
 for i in range(top50.shape[1]):
     for j in range(top50.shape[1]):
         corr_rbo = rbo(top50.iloc[:, i], top50.iloc[:, j], p=0.9)
         corr_rbo_matrix[i][j] = corr_rbo['ext']
         corr_rbo_matrix[j][i] = corr_rbo['ext']
# # rrrrr = rbo(xaa,yaa,p=0.9)
print(corr_rbo_matrix)


#%% downstream
#
colon_bg = pd.DataFrame({
    'index_in_analysis': list(range(len(colon_cl))),
    'sample_name': colon_cl

})
colon_bg.to_csv("colon_bg.txt")
dic = dict(zip(colon_bg.index_in_analysis, colon_bg.sample_name))
corr_rbo_matrix.rename(index = dic, columns = dic)
# corr_withname = corr_rbo_matrix.index.replace(colon_bg.iloc[:,0],colon_bg.iloc[:,1])
# #%% plot heatmap

plt.figure(figsize=(24,20))
sns.clustermap(
    corr_rbo_matrix,
    cmap='YlGnBu',
    # annot=True,
    linewidths=2
)
plt.show()
# %%




from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(random_forest.estimators_[103], out_file='tree104.dot',
                feature_names = featurelist,
                rounded = True, proportion = False,
                precision = 2, filled = True)





#%%
import matplotlib.pyplot as plt

correlations = falcontogo.corr()
plt.matshow(falcontogo.iloc[:, 1:len(falcontogo.columns)].corr())
plt.show()

sol = (
    correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))




















