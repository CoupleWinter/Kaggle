# coding =utf-8
# Author : Noctis
# Date : 2018-1-23 21:44 sy


import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


class GetData(object):
    path = os.path.dirname(os.path.abspath(__file__)).split('Titanic')[0] + 'Titanic/data/'
    train = path + 'train.csv'
    test = path + 'test.csv'
    submission = path + ''
    data = pd.read_csv(train)

    def pylt(self):
        """

        :return:
        """

        # fig = plt.figure()
        # 确认图表颜色
        # fig.set(alpha=0.2)
        # 分成两个图
        plt.subplot2grid((2, 3), (0, 0))
        # plt.subplots_adjust()
        # 住装图
        self.data.Survived.value_counts().plot(kind='bar')
        plt.title('Live')
        plt.ylabel('Human')

        plt.subplot2grid((2, 3), (0, 1))
        self.data.Pclass.value_counts().plot(kind='bar')
        plt.ylabel(u'Human')
        plt.title(u'Level')

        plt.subplot2grid((2, 3), (0, 2))
        plt.scatter(self.data.Survived, self.data.Age)
        plt.ylabel('Age')
        plt.grid(b=True, which='major', axis='y')
        plt.title('Cover age')

        plt.subplot2grid((2, 3), (1, 0), colspan=2)
        self.data.Age[self.data.Pclass == 1].plot(kind='kde')
        self.data.Age[self.data.Pclass == 2].plot(kind='kde')
        self.data.Age[self.data.Pclass == 3].plot(kind='kde')
        plt.xlabel('Age')
        plt.ylabel('Midu')
        plt.title('Human level')
        plt.legend(('1', '2', '3'), loc='best')

        plt.subplot2grid((2, 3), (1, 2))
        self.data.Embarked.value_counts().plot(kind='bar')
        plt.title('People in port')
        plt.ylabel('Human')

        plt.show()

    def feature_engineering(self):
        print(self.data.describe())
        self.data = self.data.fillna(0)
        self.data['Sex'] = self.data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
        data_x = self.data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'PassengerId']].as_matrix()
        data_y = self.data[['Survived']].as_matrix()
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, random_state=42)
        print(len(x_train), len(y_train), len(x_test), len(y_test))
        return x_train, x_test, y_train, y_test

    def feature_engineering_test(self):
        data = pd.read_csv(self.test)
        data = data.fillna(0)
        data['Sex'] = self.data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
        data_x = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'PassengerId']].as_matrix()
        return data, data_x


def get_data():
    path = os.path.dirname(os.path.abspath(__file__)).split('Titanic')[0] + 'Titanic/data/'
    train = path + 'train.csv'

    data = pd.read_csv(train)
    print(data.info())

    print(data.describe())
    return data
    # 标签转换(one-hot encoding)

    # dataframe['Sex'] = dataframe['Sex'].apply(lambda s: 1 if s == 'male' else 0)
    # # 填充为零
    # dataframe = dataframe.fillna(0)
    # dataframe_x = dataframe[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']]
    # dataframe_x = dataframe_x.as_matrix()
    #
    # dataframe['Deceased'] = dataframe['Survived'].apply(lambda s: int(not s))
    # dataframe_Y = dataframe[['Deceased', 'Survived']]
    # dataframe_Y = dataframe_Y.as_matrix()
    #
    # # 切分数据集,验证数据占20%
    # x_train, x_test, y_train, y_test = train_test_split(dataframe_x, dataframe_Y, test_size=0.2, random_state=42)
    # return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    GetData().feature_engineering()
