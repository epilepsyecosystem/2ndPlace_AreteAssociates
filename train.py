import pdb
import numpy as np
import os
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
import random
from sklearn.ensemble import ExtraTreesClassifier
import csv
import matplotlib as plt
import pickle
import json

settings = json.load(open('SETTINGS.json'))
pat = settings['pat']
data = pd.read_csv(settings['feat']+'/pat_'+str(pat)+'_short_train.csv')
test = pd.read_csv(settings['feat']+'/pat_'+str(pat)+'_short_test.csv')
data2 = pd.read_csv(settings['feat']+'/pat_'+str(pat)+'_long_train.csv')
test2 = pd.read_csv(settings['feat']+'/pat_'+str(pat)+'_long_test.csv')

data = pd.concat([data,data2], axis=1)
test = pd.concat([test,test2], axis=1)

# clean the training data by removing nans
data.dropna(thresh=data.shape[1]-3, inplace=True)

data.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)

data.fillna(0, inplace=True)
test.fillna(0, inplace=True)

data_file = data.File.values
test_file = test.File.values

# get labels
labela=[int(((str(os.path.basename(n)).split('_'))[2]).split('.')[0]) for n in data_file[:,0]]
labelt=[int(((str(os.path.basename(n)).split('_'))[2]).split('.')[0]) for n in test_file[:,0]]

data['L'] = labela
test['L'] = labelt

data.sort_values(['L'], inplace=True, ascending=False)
test.sort_values(['L'], inplace=True, ascending=False)

labela = data.L.values
labelt = test.L.values

data_feat = data.drop(['File', 'pat', 'Unnamed: 0', 'L'], axis=1)
test_feat = test.drop(['File', 'pat', 'Unnamed: 0', 'L'], axis=1)
feat_names = data_feat.columns
data_feat = data_feat.values
test_feat = test_feat.values

# generate model using ExtraTrees
if pat == 2:
    clf = ExtraTreesClassifier(n_estimators=5000, random_state=0, max_depth=15, n_jobs=2,criterion='entropy', min_samples_split=7)
elif pat == 3:
    clf = ExtraTreesClassifier(n_estimators=4500, random_state=0, max_depth=15,criterion='entropy', n_jobs=2)
elif pat == 1:
    clf = ExtraTreesClassifier(n_estimators=3000, random_state=0, max_depth=11, n_jobs=2)

clf.fit(data_feat, labela)
y_pred = clf.predict_proba(test_feat)

# check hold-out set
this_AUC = metrics.roc_auc_score(labelt, y_pred[:,1])
print "AUC: " + str(this_AUC)

pickle.dump(clf, open(settings['model']+'/modeldump_'+str(pat)+'_ef.pkl', 'wb'))

