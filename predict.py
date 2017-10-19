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
clf = pickle.load(open(settings['model']+'/modeldump_'+str(pat)+'_ef.pkl','rb'))

# get the test for submission
test = pd.read_csv(settings['feat']+'/pat_'+str(pat)+'_short_newtest_sub.csv')
test.sort_values(['File'], inplace=True)

test2 = pd.read_csv(settings['feat']+'/pat_'+str(pat)+'_long_newtest_sub.csv')
test2.sort_values(['File'], inplace=True)

test = pd.concat([test,test2], axis=1)

test.replace([np.inf, -np.inf], np.nan, inplace=True)

test.fillna(0, inplace=True)

test2_file = (test.File.values)[:,0]

test2_feat = test.drop(['File', 'pat', 'Unnamed: 0'], axis=1)

test2_feat = test2_feat.values

y_pred = clf.predict_proba(test2_feat)

outFile = open(settings['sub']+'/pat_' + str(pat) + '_fullmerge_2ndplace_sub.csv',"wb")
csv_writer = csv.writer(outFile)
final = y_pred[:,1]
csv_writer.writerows(zip(test2_file,final))
outFile.close() 
