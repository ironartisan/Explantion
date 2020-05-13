import os,sys
import csv
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import xgboost
import lime.lime_tabular
from common.constant import *
from utils.base import pd_to_np

class InsuranceDataset():
    """
    handle data && load data
    """

    def __init__(self, dirpath=None):
        self._dirpath = dirpath
        if not self._dirpath:
            self._dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../data', 'insurance_data')

    def load_insurance(self, data_train='carInsurance_train.csv') :
        """
        handle and load data,Only apply  to random forests.
        """
        self._train_filepath = os.path.join(self._dirpath, data_train)

        try:
            train_data = pd.read_csv(self._train_filepath)

            # define car insurance as target
            target_name = train_data.columns[-1]

            # define useful features
            feature_names = ["Age", "Job", "Marital", "Education","Balance","HHInsurance","CarLoan",
                             "NoOfContacts","DaysPassed","PrevAttempts","Outcome"]
            # define feature transform dict
            trans_dicts = {
                'Job': {
                    'blue-collar': 1,
                    'self-employed': 2,
                    'services':3,
                    'housemaid':4,
                    'entrepreneur':5,
                    'student':6,
                    'unemployed':7,
                    'technician':8,
                    'admin.':9,
                    'management':10,
                    'retired':11
                },
                'Education': {
                    'secondary': 1,
                    'tertiary': 2,
                    'primary': 3
                },
                'Marital': {
                    'Single': 1,
                    'Married': 2,
                    'Others': 3

                },
                'Outcome': {
                    'success': 1,
                    'failure': 2,
                    'other':3
                }
            }

            for feature_name in trans_dicts.keys():
                train_data[feature_name] = train_data[feature_name].map(trans_dicts[feature_name])

            x_train = train_data[feature_names]
            y_train = train_data[target_name]

            # handle NAN data
            x_train = x_train.fillna(-3)

            return x_train, y_train, feature_names, target_name
        except Exception as err:
            print("Exception: {}".format(err))
            sys.exit(1)

    def load(self, data_train=CSV_INSURANCE):
        self._train_filepath = os.path.join(self._dirpath, data_train)
        # define all features

        data = pd.read_csv(self._train_filepath)
        data = data[FEATURE_ALL]

        # use U to fill NAN
        for category in CATEGORICAL:
            data[category].fillna("U",inplace=True)

        data = pd_to_np(data)

        # encode lable car insurance
        labels = data[:, -1]

        le = sklearn.preprocessing.LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        class_names = le.classes_
        # get rid of the last column
        data = data[:, :-1]

        # lableEncode nonnumeric features
        categorical_names = {}
        for feature in CATEGORICAL_FEATURES:
            le = sklearn.preprocessing.LabelEncoder()
            le.fit(data[:, feature])
            data[:, feature] = le.transform(data[:, feature])
            categorical_names[feature] = le.classes_

        data = data.astype(float)

        # Ensure that the generated random Numbers are predictable
        np.random.seed(1)
        return data, labels, class_names, categorical_names





