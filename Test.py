import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

########## LAYER DEFINITION START ##########
'''
learning_rate = 0.01
training_epochs = 5
batch_size = 256
examples_to_show = 10

n_input = 784  # MNIST data input (img shape: 28*28)
n_hidden_1 = 256
n_hidden_2 = 128

weights = {
    'hidden1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'hidden2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
}
biases = {
    'input': tf.Variable(tf.random_normal([n_input])),
    'hidden1': tf.Variable(tf.random_normal([n_hidden_1])),
    'hidden2': tf.Variable(tf.random_normal([n_hidden_2]))
}

x = tf.placeholder("float", [None, n_input])

hidden_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['hidden1']),
                                   biases['hidden1']))
hidden_2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden_1, weights['hidden2']),
                                   biases['hidden2']))

re_hidden_1 = tf.nn.sigmoid(tf.add(tf.matmul(
             hidden_2, tf.transpose(weights['hidden2']) ),biases['hidden1']))
re_hidden_2 = tf.nn.sigmoid(tf.add(tf.matmul(
             re_hidden_1, tf.transpose(weights['hidden1']) ),biases['input']))

y_pred = re_hidden_2


y_true = x

cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
'''
########## LAYER DEFINITION END ##########

########## LAYER DEFINITION START ##########

learning_rate = 0.01
training_epochs = 5
batch_size = 256
examples_to_show = 10

n_input = 784  # MNIST data input (img shape: 28*28)
n_hidden_1 = 5
n_hidden_2 = 128
kernel_size = 3

weights = {
    'hidden1': tf.Variable(tf.random_normal([3,3,1,5])),
    'hidden2': tf.Variable(tf.random_normal([3,3,5,10])),
    'de_hidden2': tf.Variable(tf.random_normal([3,3,10,5])),
    'de_hidden1': tf.Variable(tf.random_normal([3,3,5,1]))
}
biases = {
    'input': tf.Variable(tf.random_normal([1])),
    'hidden1': tf.Variable(tf.random_normal([5])),
    'hidden2': tf.Variable(tf.random_normal([10]))
}

X = tf.placeholder("float", [None, n_input])
x = tf.reshape(X, [-1, 28, 28, 1])

hidden_1 = tf.nn.sigmoid(tf.add(tf.nn.conv2d(x, weights['hidden1'], strides=[1,1,1,1], padding='SAME'), 
                                   biases['hidden1']))
hidden_2 = tf.nn.sigmoid(tf.add(tf.nn.conv2d(hidden_1, weights['hidden2'], strides=[1,1,1,1], padding='SAME'),
                                   biases['hidden2']))

re_hidden_1 = tf.nn.sigmoid(tf.add(tf.nn.conv2d(
             hidden_2, weights['de_hidden2'], strides=[1,1,1,1], padding='SAME' ),biases['hidden1']))
re_hidden_2 = tf.nn.sigmoid(tf.add(tf.nn.conv2d(
             re_hidden_1, weights['de_hidden1'], strides=[1,1,1,1], padding='SAME' ),biases['input']))

y_pred = re_hidden_2


y_true = x

cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

########## LAYER DEFINITION END ##########


# start training
sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("logs/", sess.graph)
sess.run(tf.initialize_all_variables())


total_batch = int(mnist.train.num_examples/batch_size)
for epoch in range(training_epochs):
    # Loop over all batches
    print('epoch'+str(epoch)+':')
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        print('----iterator'+str(i)+': '+str(c))

encode = sess.run(y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode[i], (28, 28)))
plt.show()    
    
sess.close()


