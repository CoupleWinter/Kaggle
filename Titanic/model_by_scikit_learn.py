# coding : utf-8
# Author : Noctis
# Date : 2018-3-18

from Titanic.utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os

x_train, y_train, train_x, data = GetData.feature_engineering(GetData().train)

logistic_model = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

logistic_model.fit(x_train, y_train.ravel())

# Croess validation
GetData.cross_validation(x_train, y_train.ravel(), data, logistic_model)

# y_pred = logistic_model.predict(x_test)
#
# print(accuracy_score(y_test, y_pred))

GetData.trans_model_to_feature(train_x, logistic_model)

data, x_submission = GetData.feature_engineering_test(GetData().test)

y_submission = logistic_model.predict(x_submission)

result = pd.DataFrame(
        {'PassengerId': data['PassengerId'].as_matrix(), 'Survived': y_submission.astype(np.int32)})
path = os.path.dirname(os.path.abspath(__file__)).split('Titanic')[0] + 'Titanic/data/'
result.to_csv(path + 'logistic_regression_predictions.csv', index=False)
