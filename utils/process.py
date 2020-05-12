#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : cyl
# @Time : 2020/5/11 16:45 
import sklearn
import sklearn.ensemble
import numpy as np
from utils.insurance_dataset import InsuranceDataset
import lime.lime_tabular
import pandas as pd

def pd_to_np(pd_data):
    return pd.DataFrame.to_numpy(pd_data)

def process():
    insur_dataset = InsuranceDataset()
    data, labels, feature_names, target_name = insur_dataset.load_insurance()
    train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.80)

    # use RandomForestClassifier
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)

    # fit
    rf.fit(train, labels_train)
    # predict output
    print("accuracy score is ",sklearn.metrics.accuracy_score(labels_test, rf.predict(test)))

    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=pd_to_np(train), feature_names=feature_names)
    test = pd_to_np(test)
    # pick one at random
    i = np.random.randint(0, test.shape[0])
    exp = explainer.explain_instance(test[i], rf.predict_proba, num_features=10, top_labels=2)

    return exp.as_list()

def xgb_process():
    insur_dataset = InsuranceDataset()
    insur_dataset.load()

if __name__=="__main__":
    xgb_process()

