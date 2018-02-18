# coding=utf-8


import os
import pandas as pd
import tensorflow as tf

testdata = pd.read_csv('data/test.csv')
print(testdata.info())
testdata = testdata.fillna(0)
testdata['Sex'] = testdata['Sex'].apply(lambda s: 1 if s == 'male' else 0)
X_test = testdata[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']].as_matrix()

session = tf.InteractiveSession()
path = os.getcwd()
savepath = path + ''
