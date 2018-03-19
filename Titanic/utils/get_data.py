# coding =utf-8
# Author : Noctis
# Date : 2018-1-23 21:44 sy


import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor


class GetData(object):
    path = os.path.dirname(os.path.abspath(__file__)).split('Titanic')[0] + 'Titanic/data/'
    train = path + 'train.csv'
    test = path + 'test.csv'
    submission = path + ''

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

    @staticmethod
    def feature_engineering(path):
        """

        :param path:
        :return:
        """
        data = pd.read_csv(path)
        print(data.describe())
        # self.data = self.data.fillna(0)

        data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)

        # region 均值归一
        scaler = preprocessing.StandardScaler()
        # endregion
        data_x = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'PassengerId']]
        data_y = data[['Survived']]

        age_scale_param = scaler.fit(data_x[['Age']])
        data[['Age']] = scaler.fit_transform(data_x[['Age']], age_scale_param)
        fare_scale_param = scaler.fit(data[['Fare']])
        data[['Fare']] = scaler.fit_transform(data[['Fare']], fare_scale_param)
        print(data)
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, random_state=42)
        print(len(x_train), len(y_train), len(x_test), len(y_test))
        return x_train, x_test, y_train, y_test

    def feature_engineering_test(self):
        data = pd.read_csv(self.test)
        data = data.fillna(0)
        data['Sex'] = self.data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
        data_x = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'PassengerId']].as_matrix()
        return data, data_x

    @staticmethod
    def set_missing_ages(path):
        """
        目的是对缺少的特性进行补充
        方式：对已有特性按照年龄分组，使用已有特性进行训练模型，对缺少字段进行预测
        算法： 随机森林
        :return:
        """
        data = pd.read_csv(path)
        data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
        age_data = data[['Age', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Fare', 'PassengerId']]
        know_age_data = age_data[age_data.Age.notnull()].as_matrix()
        un_know_age_data = age_data[age_data.Age.isnull()].as_matrix()
        y = know_age_data[:, 0]
        X = know_age_data[:, 1:]
        rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
        rfr.fit(X, y)
        predicte_ages = rfr.predict(un_know_age_data[:, 1::])
        data.loc[(data.Age.isnull()), 'Age'] = predicte_ages
        return data, rfr,

    @staticmethod
    def set_cabin_type(data):
        """
        原本一个属性维度，因为其取值可以是[‘yes’,’no’]，而将其平展开为’Cabin_yes’,’Cabin_no’两个属性
        :param data:
        :return:
        """
        data.loc[(data.Cabin.notnull()), 'Cabin'] = 'Yes'
        data.loc[(data.Cabin.isnull()), 'Cabin'] = 'No'
        return data

    @staticmethod
    def get_dummies(data):
        """
        类目型的特征因子化
        :param data:
        :return:
        """
        dummies_Cabin = pd.get_dummies(data['Cabin'], prefix='Cabin')
        dummies_Embarked = pd.get_dummies(data['Embarked'], prefix='Embarked')
        dummies_Sex = pd.get_dummies(data['Sex'], prefix='Sex')
        dummies_Pclass = pd.get_dummies(data['Pclass'], prefix='Pclass')
        df = pd.concat([data, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
        df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
        return df


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
    train_path = GetData().train
    data, rfr = GetData.set_missing_ages(train_path)
    data = GetData.set_cabin_type(data)
    df = GetData.get_dummies(data)
    print(df)
