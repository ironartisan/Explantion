import os,sys
import csv
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import xgboost
import lime.lime_tabular

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
        handle data && load data
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

    def load(self, data_train='carInsurance_train.csv'):
        self._train_filepath = os.path.join(self._dirpath, data_train)
        # define all features
        feature_all = ["Age", "Job", "Marital", "Education", "Balance", "HHInsurance", "CarLoan",
                         "NoOfContacts", "DaysPassed", "PrevAttempts", "Outcome","CarInsurance"]
        categorical = ["Job", "Marital", "Education","Outcome"]

        data = pd.read_csv(self._train_filepath)
        data = data[feature_all]
        # use U to fill NAN
        for category in categorical:
            data[category].fillna("U",inplace=True)
        # print(data.isnull().sum())

        data = pd.DataFrame.to_numpy(data)

        # encode lable car insurance
        labels = data[:, -1]

        le = sklearn.preprocessing.LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        class_names = le.classes_
        # get rid of the last column
        data = data[:, :-1]

        # define useful features
        feature_names = ["Age", "Job", "Marital", "Education", "Balance", "HHInsurance", "CarLoan",
                         "NoOfContacts", "DaysPassed", "PrevAttempts", "Outcome"]

        # Transform nonnumeric features "Job", "Marital", "Education","Outcome"
        categorical_features = [1, 2, 3, 10]

        # lableEncode nonnumeric features
        categorical_names = {}
        for feature in categorical_features:
            le = sklearn.preprocessing.LabelEncoder()
            le.fit(data[:, feature])
            data[:, feature] = le.transform(data[:, feature])
            categorical_names[feature] = le.classes_

        data = data.astype(float)
        #  encode nonnumeric features by one-hot
        encoder = sklearn.preprocessing.OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore')

        # Ensure that the generated random Numbers are predictable
        np.random.seed(1)
        train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.80)

        encoder.fit(data)
        encoded_train = encoder.transform(train)

        # use gradient boosted trees as the model
        gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
        gbtree.fit(encoded_train, labels_train)
        # accuracy score
        acc_score = sklearn.metrics.accuracy_score(labels_test, gbtree.predict(encoder.transform(test)))
        # print(acc_score)
        predict_fn = lambda x: gbtree.predict_proba(encoder.transform(x)).astype(float)

        explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=feature_names, class_names=class_names,
                                                           categorical_features=categorical_features,
                                                           categorical_names=categorical_names,kernel_width=3)
        np.random.seed(1)

        i = np.random.randint(0, test.shape[0])
        exp = explainer.explain_instance(test[i], predict_fn, num_features=5)
        return exp.as_list()



