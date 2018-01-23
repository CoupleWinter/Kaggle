# coding =utf-8
# Author : Noctis
# Date : 2018-1-23 21:44 sy


import tensorflow as tf
from tqdm import tqdm

session = tf.InteractiveSession()

x = tf.placeholder(dtype=tf.float32, shape=[None, 6], name='feature')
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='y')

w = tf.Variable(initial_value=tf.random_normal([6, 2]), name='weight')
b = tf.Variable(initial_value=tf.random_normal([2]), name='bias')

cross = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=tf.matmul(x * w) + b, )
cross_entropy = tf.reduce_mean(cross)
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

tf.global_variables_initializer().run()

for x in tqdm(10):
    feed = {x: '', y_: ''}
    _, loss = session.run([train_op, cross_entropy], feed=feed)

if __name__ == '__main__':
    print(x)
