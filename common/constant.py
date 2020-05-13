#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : cyl
# @Time : 2020/5/13 8:30

# all features
FEATURE_ALL= ["Age", "Job", "Marital", "Education", "Balance", "HHInsurance", "CarLoan",
               "NoOfContacts", "DaysPassed", "PrevAttempts", "Outcome", "CarInsurance"]
# nonnumeric features
CATEGORICAL= ["Job", "Marital", "Education", "Outcome"]

# useful features
FEATURE_NAMES = ["Age", "Job", "Marital", "Education", "Balance", "HHInsurance", "CarLoan",
                 "NoOfContacts", "DaysPassed", "PrevAttempts", "Outcome"]

#  nonnumeric features ("Job", "Marital", "Education","Outcome") index
CATEGORICAL_FEATURES = [1, 2, 3, 10]

CSV_INSURANCE = "carInsurance_train.csv"