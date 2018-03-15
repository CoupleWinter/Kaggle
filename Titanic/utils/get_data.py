# coding =utf-8
# Author : Noctis
# Date : 2018-1-23 21:44 sy


import os
import sys
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split


def get_data():
    path = os.path.dirname(os.path.abspath(__file__)).split('Titanic')[0] + 'Titanic/data/'
    train = path + 'train.csv'
    if not tf.gfile.Exists(train):
        print('No data')

    dataframe = pd.read_csv(train)
    print(dataframe.info())
    # 标签转换(one-hot encoding)

    dataframe['Sex'] = dataframe['Sex'].apply(lambda s: 1 if s == 'male' else 0)
    # 填充为零
    dataframe = dataframe.fillna(0)
    dataframe_x = dataframe[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
    dataframe_x = dataframe_x.as_matrix()

    dataframe['Deceased'] = dataframe['Survived'].apply(lambda s: int(not s))
    dataframe_Y = dataframe[['Deceased', 'Survived']]
    dataframe_Y = dataframe_Y.as_matrix()

    # 切分数据集,验证数据占20%
    x_train, x_test, y_train, y_test = train_test_split(dataframe_x, dataframe_Y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test
