# coding = utf-8

# import load_iris from  dataset module
from sklearn.datasets import load_iris
# Classification accuracy
from sklearn import metrics

# save "bunch" object containing iris dataset and its attributes
iris = load_iris()
###############
# print(type(iris))
#
# # print iris data
# print(iris.data)
#
# # 四个特征的名称
# print(iris.feature_names)
#
# # print integers representing the species of each observation
# print(iris.target)
#
# # print the encoding scheme for species: 0 = setosa, 1 = versicolor, 2 = virginica
# print(iris.target_names)
#
# print(iris.data.shape)
###############
X = iris.data
y = iris.target
# region Agenda 2018-3-14
#  1.K-nearest neighbors classification model?
#  2.Four steps for model training and prediction in scikit-learn?

# Step 1 : import the class
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Step 2 : Instantiate
knn = KNeighborsClassifier(n_neighbors=1)
print(knn)
knn_logis = LogisticRegression()
print(knn_logis)
# Step 3 : Fit the model with data
#
knn.fit(X, y)

knn_logis.fit(X, y)

# Step 4 : Predict observations are called "out-of-sample" data
y_ne = knn.predict(X)
print(metrics.accuracy_score(y, y_ne))

y_pred = knn_logis.predict(X)

print(metrics.accuracy_score(y, y_pred))
#  3.How can i apply this pattern to other machine learning models?
# endregion
