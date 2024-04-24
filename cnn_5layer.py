#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 03:07:22 2018

@author: sophialiu
"""

import numpy as np  # mathematical operations
import tensorflow as tf  # For graphical operations
import datetime
from sklearn.model_selection import train_test_split

IMG_SIZE = 128

x_matrix = np.load('train_data_x.npy')
y_brand = np.load('train_data_y_brand_one_hot_224.npy')
#y_product = np.load('test_data_224_y_product_one_hot.npy')
#y_productBrand = np.load('test_data_224_y_productBrand_one_hot.npy')

train_cv_x, test_x, train_cv_y, test_y = train_test_split(x_matrix, y_brand, 
                                        random_state=1, 
                                        train_size=0.7)

X, cv_x, Y, cv_y = train_test_split(train_cv_x, train_cv_y, 
                                        random_state=1, 
                                        train_size=0.7)

print('X and Y shape is: ', X.shape, Y.shape)

print('cvX and cvY shape is: ', cv_x.shape, cv_y.shape)

print('testX and testY shape is: ', test_x.shape, test_y.shape)

# STARTING WEIGHTS
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

# Starting BIAS
def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

# Function for convolution operation
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Function for pooling layer
def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Function for convolution layer
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

# Function for FC layer
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

# Resetting graph
# tf.reset_default_graph()

epochs = 100

# Defining Placeholders
x = tf.placeholder(tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, 3])
print(x.shape)
y_true = tf.placeholder(tf.float32, shape=[None, 7])
x = tf.reshape(x, [-1, IMG_SIZE, IMG_SIZE, 3])
x = x / 255.0
print(x.shape)

with tf.name_scope('Model'):
    # Convolutional Layer 1 + RELU
    convo_1 = convolutional_layer(x, shape=[10, 10, 3, 8])
    print(convo_1.shape)
    # Pooling Layer 1
    convo_1_pooling = max_pool_2by2(convo_1)
    print(convo_1_pooling.shape)
    # Convolutional Layer 2 + RELU
    convo_2 = convolutional_layer(convo_1_pooling, shape=[8, 8, 8, 16])
    print(convo_2.shape)
    # Pooling Layer 2
    convo_2_pooling = max_pool_2by2(convo_2)
    print(convo_2_pooling.shape)
    # Convolutional Layer 3 + RELU
    convo_3 = convolutional_layer(convo_2_pooling, shape=[6, 6, 16, 24])
    # Pooling Layer 3
    convo_3_pooling = max_pool_2by2(convo_3)
    # Convolutional Layer 4 + RELU
    convo_4 = convolutional_layer(convo_3_pooling, shape=[4, 4, 24, 32])
    # Pooling Layer 4
    convo_4_pooling = max_pool_2by2(convo_4)
    # Flattening
    convo_4_flat = tf.reshape(convo_1_pooling, [-1, 16 * 16 * IMG_SIZE])
    print(convo_4_flat.shape)
    # Fully Connected 1 + RELU
    full_layer_one = tf.nn.relu(normal_full_layer(convo_4_flat, IMG_SIZE))
    print(full_layer_one.shape)
    # Dropout Layer 1
    hold_prob1 = tf.placeholder(tf.float32)
    full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob1)
    print('dropout one shape: ', full_one_dropout.shape)
    # Fully Connected 1 + RELU
    full_layer_two = tf.nn.relu(normal_full_layer(full_one_dropout, IMG_SIZE))
    # Dropout Layer 1
    hold_prob2 = tf.placeholder(tf.float32)
    full_two_dropout = tf.nn.dropout(full_layer_two, keep_prob=hold_prob2)
    print('dropout two shape: ', full_two_dropout.shape)
    # Output Layer,containing 2 output nodes.
    y_pred = normal_full_layer(full_two_dropout, 7)
    print(y_pred.shape)
    # y_pred = tf.reshape(y_pred, [7, 2])

# Defining loss function
with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

# Defining objective
with tf.name_scope('ADAM'):
    train = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cross_entropy)

# Defining Accuracy
with tf.name_scope('Accuracy'):
    matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    acc = tf.reduce_mean(tf.cast(matches, tf.float32))

# Initializing weights
init = tf.global_variables_initializer()

# Tensorboard
tf.summary.scalar("loss", cross_entropy)
tf.summary.scalar("accuracy", acc)
merged_summary_op = tf.summary.merge_all()

# Starting Empty lists to keep results
acc_list = []
cross_entropy_list = []
acc_train = []
# saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    # summary_writer = tf.summary.FileWriter(TRAIN_DIR, graph=tf.get_default_graph())
    for i in range(epochs):
        # for j in range(0, steps, step_size):
        # Feeding step_size-amount data with 0.8 keeping probabilities on DROPOUT LAYERS
        _, c, summary, d = sess.run([train, cross_entropy, merged_summary_op, acc], feed_dict={x: X, y_true: Y, hold_prob1: 0.8, hold_prob2: 0.8})
        # summary_writer.add_summary(summary, i )
        acc_train.append(d)
            # Calculating CV loss and CV accuracy
        mean_of_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_true: Y, hold_prob1: 1.0, hold_prob2: 1.0})
        mean_of_acc = sess.run(acc, feed_dict={x: X, y_true: Y, hold_prob1: 1.0, hold_prob2: 1.0})
            # Appending loss and accuracy
        cross_entropy_list.append(mean_of_cross_entropy)
        acc_list.append(mean_of_acc)
        print("Iteration: [%4d/%4d], ce_loss: %.4f, accuracy: %.4f" % (i, epochs, mean_of_cross_entropy, mean_of_acc))
    # Saving the model
    # saver.save(sess, os.getcwd() + "\\CNN_BI.ckpt")
    # Printing test accuracy and cross entropy loss on test data.
    print("test accuracy = ", sess.run(acc, feed_dict={x: test_x, y_true: test_y, hold_prob1: 1.0, hold_prob2: 1.0}))
    print("cross_entropy loss = ", sess.run(cross_entropy, feed_dict={x: test_x, y_true: test_y, hold_prob1: 1.0, hold_prob2: 1.0}))

    # save model
    string_time = datetime.datetime.now().strftime("%I%M%p_%B_%d_%Y")

    str_temp = 'step_100_' + string_time + '.npy'
    save_dict = {var.name: var.eval(sess) for var in tf.global_variables()}
    np.save(str_temp, save_dict)
