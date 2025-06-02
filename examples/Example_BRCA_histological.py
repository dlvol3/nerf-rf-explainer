# An instance of NERF
# Using TCGA breast cancer dataset, mRNA=>histological classes
# NERF V0.2.1
# Yue Zhang <yue.zhang@lih.lu>



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
import platform
import time
import mygene
# NewtworkX
import math
import networkx as nx

if platform.system() == 'Windows':
    # Windows in the lab
    B20000 = pd.read_table("P:/VM/TCGA/Data/BRCA/nosmote.csv", sep=',')
if platform.system() == 'Darwin':
    # My mac
    B20000 = pd.read_table("/Users/yue/Pyc/NERF-RF_interpreter/data/BMReady.txt", sep='\t')
# Read in data and display first 5 rows

# Creating the dependent variable class
factor = pd.factorize(B20000['histological_type'])
B20000.histological_type = factor[0]
definitions = factor[1]
x = B20000.iloc[:, 1:20530].values   # Features for training
y = B20000.iloc[:, 0].values  # Labels of training

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=123)

rf1 = RandomForestClassifier(n_estimators=1000, criterion='gini', max_features="sqrt",
                             oob_score=True, n_jobs=13, max_depth=12,
                             verbose=0)




rf1.fit(x_train, y_train)

featurelistBR = B20000.iloc[:, 1:20530].columns.values.tolist()
indexBR = list(range(len(featurelistBR)))
rf1.predict(x_test)
ff_his = flatforest(rf1, x_test)
nt_his = nerftab(ff_his)
#%% # case
t5 = localnerf(nt_his, 5)
BMresult = twonets(t5, "BM1000trees1", indexBR, featurelistBR)
#%% # case 5
t4 = localnerf(nt_his, 4)
BMresult = twonets(t4, "BM1000trees0", indexBR, featurelistBR)
#%% # case 5
t6 = localnerf(nt_his, 6)
BMresult = twonets(t6, "BM1000trees1-2", indexBR, featurelistBR)

#%%
# Feature importance list

feature_importance_values = rf1.feature_importances_
feature_importances = pd.DataFrame({'feature': featurelistBR, 'importance': feature_importance_values})
feature_importances.to_csv(os.getcwd() + '/output/featureimp1000.txt', sep='\t')

#%%
ff_his = flatforest(rf1, x_test)
nt_his = nerftab(ff_his)
t1 = localnerf(nt_his, 1)
t2 = localnerf(nt_his, 2)
t1.to_csv('testoutput_his1.txt', sep='\t')
t2.to_csv('testoutput_his0.txt', sep='\t')

# Lap example
#%%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification

import platform
#%%
if platform.system() == 'Windows':
    # Windows in the lab
    gdscic = pd.read_csv('P:/VM/Drug/data/output/GDSCIC50.csv')
    ccleic = pd.read_csv('P:/VM/Drug/data/output/CCLEIC50.csv')
if platform.system() == 'Darwin':
    # My mac
    gdscic = pd.read_csv('/Users/yue/Pyc/Drug2018Sep/data/GDSCIC50.csv')
    ccleic = pd.read_csv('/Users/yue/Pyc/Drug2018Sep/data/CCLEIC50.csv')

# Extract the Lat. Drug data from both of the datasets
gdscic.head(5)
lapagdsc = gdscic.loc[(gdscic.drug == 'Lapatinib')]
lapaccle = ccleic.loc[(ccleic.drug == 'Lapatinib')]

# Create list for subset
ciLapa = list(range(3, len(lapaccle.columns), 1))
ciLapa.insert(0, 1)

# subset two sets
lapaC = lapaccle.iloc[:, ciLapa]
lapaG = lapagdsc.iloc[:, ciLapa]
lapaC.head(1)
#%%
# Branch: check the correlated features of ERBB2




#%%
# -------------------------###
# Get familiar with python DS
# ref: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

# Label the class
le = LabelEncoder()
le_count = 0

# iterate through columns
for col in lapaC:
    if lapaC.loc[:, col].dtype == 'object':
        # if less than 2 classes(which is better to use one-hot coding if not)
        if len(list(lapaC.loc[:, col].unique())) <= 2:
            # 'train' the label encoder with the training data
            le.fit(lapaC.loc[:, col])
            # Transform both training and testing
            lapaC.loc[:, col] = le.transform(lapaC.loc[:, col])
            lapaG.loc[:, col] = le.transform(lapaG.loc[:, col])

            # Keep track of how many columns were labeled
            le_count += 1

print('%d columns were label encoded.' % le_count)

#%%
# Exploratory Data Analysis(EDA)

# Distribution of the target classes(columns)
lapaC['SENRES'].value_counts()
lapaC['SENRES'].head(4)

lapaC['SENRES'].plot.hist()
plt.show()

#%%
# Examine Missing values
def missing_value_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum()/len(df)

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

Missing_values = missing_value_table(lapaC)
Missing_values.head(10)
#%%
# Column Types
# Number of each type of column
lapaC.dtypes.value_counts()

# Check the number of the unique classes in each object column
lapaC.select_dtypes('object').apply(pd.Series.nunique, axis=0)
#%% branch: check the correlation status of ERBB2


erbb2c = lapaC.iloc[:, 2:(lapaC.shape[1]-1)].corr()
print('Most Positive Correlations:\n', erbb2c["ENSG00000141736"].tail(15))
print('\nMost Negative Correlations:\n', erbb2c["ENSG00000141736"].head(15))




#%%
# Correlations
correlations = lapaC.iloc[:, 0:200].corr()['SENRES'].sort_values(na_position='first')

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))
# Create Cross-validation and training/testing


#%%
# Random forest 1st

# Define the RF
random_forest = RandomForestClassifier(n_estimators=100, random_state=123, max_features="sqrt",
                                       criterion="gini", oob_score=True, n_jobs=10, max_depth=12,
                                       verbose=0)
#%%
# Drop SENRES

train_labels = lapaC.loc[:, "SENRES"]
cell_lines_lapaC = lapaC.loc[:, "ccle.name"]
lapaC = lapaC.drop(['ccle.name'], axis=1)

if 'SENRES' in lapaC.columns:
    train = lapaC.drop(['SENRES'], axis=1)
else:
    train = lapaC.copy()
train.iloc[0:3,0:3]
features = list(train.columns)
# train["SENRES"] = train_labels



#%%

# RF 1st train 5 trees

random_forest.fit(train, train_labels)

# Extract feature importances
feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})
feature_importances
train.shape
# Make predictions on the test data
test_labels = lapaG.loc[:, "SENRES"]
cell_lines_lapaG = lapaG.loc[:, "gdsc.name"]

if 'SENRES' in lapaG.columns:
    test = lapaG.drop(['SENRES'], axis=1)
else:
    test = lapaG.copy()

test = test.drop(['gdsc.name'], axis=1)
predictions = random_forest.predict(test)
predictions

confusion_matrix(test_labels, predictions)
#%%
random_forest.oob_score_
#
test_pred = random_forest.predict(test)
random_forest.decision_path(test)
len(list(test_pred))
print(confusion_matrix(test_labels, test_pred))
random_forest.get_params(deep=True)




#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
tree1 = random_forest.estimators_[2]
from sklearn import tree
dotdata = export_graphviz(tree1, out_file=None,
                feature_names=train.columns,
                rounded=True, proportion=False,
                precision=2,filled=True)

graph5 = graphviz.Source(dotdata)
graph5.render("test")
from subprocess import call
import os
dir_path = os.getcwd()

tree1.tree_.impurity

print(dotdata)
#%%
# Check cross-validation result
# from sklearn.model_selection import cross_val_score

# print(np.mean(cross_val_score(random_forest, train, train_labels, cv=10)))
# print(cross_val_score(random_forest, train, train_labels, cv=10))

#%%
# Module for the network approaches
# Extract the information in the decision tree

n_nodes = random_forest.estimators_[2].tree_.node_count
children_left = random_forest.estimators_[2].tree_.children_left
children_right = random_forest.estimators_[2].tree_.children_right
feature = random_forest.estimators_[2].tree_.feature
threshold = random_forest.estimators_[2].tree_.threshold
testy = train.iloc[[2, 4]]
decision_p = random_forest.decision_path(testy)
leave_p = random_forest.apply(test)
decision_p[0].indices
testy.shape
print(decision_p)


#%%
# Create feature list, convert ENSG into gene symbols
featurelist = train.columns.values.tolist()
# Mygene convertion
mg = mygene.MyGeneInfo()
mg.metadata('available_fields')
con = mg.querymany(featurelist, scopes='ensembl.gene', fields='symbol', species="human", as_dataframe=True)
# replace Nan unmapped with original ENSGZ
con['symbol'] = np.where(con['notfound'] == True, con.index.values, con['symbol'])

featurelist_g = con.iloc[:, 3].reset_index()
feag = featurelist_g.iloc[:, 1]
# featurelist_g.loc[featurelist_g['query'] == 'ENSG00000229425'].index[0]


feag.pop(47081)

# POP out those duplicates
feag = list(feag)

index = list(range(len(featurelist)))

sl = random_forest.feature_importances_
fl = pd.DataFrame({
    'feature_name': feag,
    'score': sl,
    'index': index
})


fls = fl.sort_values('score', ascending=False)

#%%


#%%


#%%
# Other cancer
# Uninarytract bladder 2, A549 lung 16, BT549 Breast 31
testy2 = train.iloc[[23, 30]]
TIE2_f = flatforest(random_forest, testy2)
TIE2 = extarget(random_forest, testy2, TIE2_f)
nt_lap2 = nerftab(TIE2)

#%% 3 cell lines
# tbla = localnerf(nt_lap2, 0)
# tlung = localnerf(nt_lap2, 1)
# tbreast = localnerf(nt_lap2, 2)


#%% SEN and RES
tau565 = localnerf(nt_lap2, 0)
t474 = localnerf(nt_lap2, 1)

AU565 = twonets(tau565, "AU565S", index, feag)
BT474 = twonets(t474, "BT549S", index, feag)
#%%
exp = random_forest.decision_path(testy2)
#%%


BT549 = twonets(tbreast, "BT549_breast", index, feag)
UT = twonets(tbla, "UT_bladder", index, feag)
A549 = twonets(tlung, "A549_lung", index, feag)
#%% # Javascript version of the network X
jsnx.draw(Glung, {
    element: '#canvas',
    weighted: true,
    edgeStyle: {
        'stroke-width': 10
    }
});


nx.write_gexf(Glung, os.getcwd() + '/output/1_lung.gexf', encoding='utf-8', prettyprint=True)
#%%
# Similarity between the two predictions

#%%
# Feature importance list

feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': featurelist, 'importance': feature_importance_values})
feature_importances.to_csv(os.getcwd() + '/output/featureimpLap_feb.txt', sep='\t')

#%%
# Grid search
# Grid search for the best Hyperpara.

np.random.seed(123)
start = time.time()

param_dist = {'max_depth': [2,6,12],
              'bootstrap': [True, False],
              'max_features': ['sqrt', 'log2', None]
              }

cv_rf = GridSearchCV(random_forest, cv=10,
                     param_grid=param_dist,
                     n_jobs = 10)

cv_rf.fit(train, train_labels)
print('Best Parameters using grid search: \n',
      cv_rf.best_params_)
end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))
#%%
# Cross-validation on the mother model
def cross_val_metrics(fit, training_set, class_set, estimator, print_results = True):
    """
    Purpose
    ----------
    Function helps automate cross validation processes while including
    option to print metrics or store in variable
    Parameters
    ----------
    fit: Fitted model
    training_set:  Data_frame containing 80% of original dataframe
    class_set:     data_frame containing the respective target vaues
                      for the training_set
    print_results: Boolean, if true prints the metrics, else saves metrics as
                      variables
    Returns
    ----------
    scores.mean(): Float representing cross validation score
    scores.std() / 2: Float representing the standard error (derived
                from cross validation score's standard deviation)
    """
    my_estimators = {
        'rf': 'estimators_',
    }
    try:
        # Captures whether first parameter is a model
        if not hasattr(fit, 'fit'):
            return print("'{0}' is not an instantiated model from scikit-learn".format(fit))

        # Captures whether the model has been trained
        if not vars(fit)[my_estimators[estimator]]:
            return print("Model does not appear to be trained.")

    except KeyError as e:
        print("'{0}' does not correspond with the appropriate key inside the estimators dictionary. \
\nPlease refer to function to check `my_estimators` dictionary.".format(estimator))
        raise

    n = KFold(n_splits=10)
    scores = cross_val_score(fit,
                             training_set,
                             class_set,
                             cv = n
                             )
    if print_results:
        for i in range(0, len(scores)):
            print("Cross validation run {0}: {1: 0.3f}".format(i, scores[i]))
        print("Accuracy: {0: 0.3f} (+/- {1: 0.3f})"\
              .format(scores.mean(), scores.std() / 2))
    else:
        return scores.mean(), scores.std() / 2

#%%
cross_val_metrics(random_forest,
                  train,
                  train_labels,
                  'rf',
                  print_results=True,
                  )
#%%
print(__doc__)

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

# #############################################################################
# Data IO and generation

# Import some data to play with
X = train
y = train_labels
X = X.values
y = y.values

n_samples, n_features = X.shape
X = x_test
y = y_test
X = X.values
y = y.values

# #############################################################################
# Classification and ROC analysis
#%%

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=6)
classifier = rf1

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic CV')
plt.legend(loc="lower right")
plt.savefig('Per.png')
plt.show()