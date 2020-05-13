#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : cyl
# @Time : 2020/5/11 16:45 
import sklearn
import sklearn.ensemble
import numpy as np
import lime.lime_tabular
import pandas as pd
import xgboost
from utils.insurance_dataset import InsuranceDataset
from utils.base import encode_onehot, split_sample
from common.constant import *


class Predict():
    def __init__(self):
        self.insur_dataset = InsuranceDataset()
        self.encode_onehot = encode_onehot()

    def pd_to_np(self, pd_data):
        return pd.DataFrame.to_numpy(pd_data)

    def rf_predict(self):
        data, labels, feature_names, target_name = self.insur_dataset.load_insurance()
        train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.80)

        # use RandomForestClassifier
        rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)

        # fit
        rf.fit(train, labels_train)
        # predict output
        print("accuracy score is ",sklearn.metrics.accuracy_score(labels_test, rf.predict(test)))

        explainer = lime.lime_tabular.LimeTabularExplainer(training_data=self.pd_to_np(train), feature_names=feature_names)
        test = self.pd_to_np(test)
        # pick one at random
        i = np.random.randint(0, test.shape[0])
        exp = explainer.explain_instance(test[i], rf.predict_proba, num_features=10, top_labels=2)
        print(exp.as_list())

        return exp.as_list()

    def xgb_predict(self):
        data, labels, class_names, categorical_names= self.insur_dataset.load()
        train, test, labels_train, labels_test = split_sample(data, labels)

        self.encode_onehot.fit(data)
        encoded_train = self.encode_onehot.transform(train)

        # use gradient boosted trees as the model
        gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
        gbtree.fit(encoded_train, labels_train)
        # accuracy score
        acc_score = sklearn.metrics.accuracy_score(labels_test, gbtree.predict(self.encode_onehot.transform(test)))
        # print(acc_score)
        predict_fn = lambda x: gbtree.predict_proba(self.encode_onehot.transform(x)).astype(float)

        explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=FEATURE_NAMES, class_names=class_names,
                                                           categorical_features=CATEGORICAL_FEATURES,
                                                           categorical_names=categorical_names, kernel_width=3)
        np.random.seed(1)
        # pick a random instance
        i = np.random.randint(0, test.shape[0])
        exp = explainer.explain_instance(test[i], predict_fn, num_features=10)
        # print(exp.as_list())

        return exp.as_list()




