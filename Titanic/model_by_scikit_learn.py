# coding : utf-8
# Author : Noctis
# Date : 2018-3-18

from Titanic.utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os

x_train, x_test, y_train, y_test = GetData.feature_engineering(GetData().train)


logistic_model = LogisticRegression()
logistic_model.fit(x_train, y_train.ravel())
y_pred = logistic_model.predict(x_test)

#
print(accuracy_score(y_test, y_pred))

data, x_submission = GetData.feature_engineering_test(GetData().test)

y_submission = logistic_model.predict(x_submission)

result = pd.DataFrame(
        {'PassengerId': data['PassengerId'].as_matrix(), 'Survived': y_submission.astype(np.int32)})
path = os.path.dirname(os.path.abspath(__file__)).split('Titanic')[0] + 'Titanic/data/'
result.to_csv(path + 'logistic_regression_predictions.csv', index=False)
