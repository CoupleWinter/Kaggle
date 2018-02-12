# coding =utf-8
# Author : Noctis
# Date : 2018-1-23 21:44 sy


import tensorflow as tf
from tqdm import tqdm

session = tf.InteractiveSession()

X = tf.placeholder(dtype=tf.float32, shape=[None, 6], name='feature')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='y')

w = tf.Variable(initial_value=tf.random_normal([6, 2]), name='weight')
b = tf.Variable(initial_value=tf.zeros([2]), name='bias')

y_pred = tf.nn.softmax(tf.matmul(X, w) + b)

cross_entropy = -tf.reduce_sum(Y * tf.log(y_pred + 1e-10), reduction_indices=1)
cost = tf.reduce_mean(cross_entropy)

train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

tf.global_variables_initializer().run()

for x in tqdm(10):
    feed = {x: '', y_pred: ''}
    _, loss = session.run([train_op, cross_entropy], feed=feed)

if __name__ == '__main__':
    print(x)
