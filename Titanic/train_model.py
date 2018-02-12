# coding =utf-8
# Author : Noctis
# Date : 2018-1-23 21:44 sy


import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/train.csv')

data = data.fillna(0)
data['Sex'] = data['Sex'].apply(lambda s: 1 if s == 'male' else 0)
data['Deceased'] = data['Survived'].apply(lambda s: 1 - s)
dataset_X = data[['Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']].as_matrix()
dataset_Y = data[['Deceased', 'Survived']].as_matrix()

# split training data and validation set data

X_train, X_val, y_train, y_val = train_test_split(dataset_X, dataset_Y, test_size=0.2, random_state=42)

session = tf.InteractiveSession()

X = tf.placeholder(dtype=tf.float32, shape=[None, 6], name='feature')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='y')

w = tf.Variable(initial_value=tf.random_normal([6, 2]), name='weight')
b = tf.Variable(initial_value=tf.zeros([2]), name='bias')

y_pred = tf.nn.softmax(tf.matmul(X, w) + b)

cross_entropy = -tf.reduce_sum(Y * tf.log(y_pred + 1e-10), reduction_indices=1)
cost = tf.reduce_mean(cross_entropy)

train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

# calculate accuracy
correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(y_pred, 1))
acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.global_variables_initializer().run()

for i in range(10):
    total_loss = 0.
    for j in range(10):
        feed_dict = {X: [X_train[i]], Y: [y_train[i]]}
        _, loss = session.run([train_op, cost], feed_dict=feed_dict)
        total_loss += loss
    print('')