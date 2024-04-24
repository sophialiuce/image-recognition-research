#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 01:28:18 2018

@author: sophialiu
"""

import tensorflow as tf
import numpy as np
import datetime
from tensorflow.python import pywrap_tensorflow
from tensorflow.contrib.slim.nets import inception
from Inception_v1_author import inception_v1
from sklearn.model_selection import train_test_split

slim = tf.contrib.slim

IMG_SIZE = 224
x_matrix = np.load('train_data_x_224.npy')
#y_brand = np.load('train_data_y_brand_one_hot_224.npy')
y_pb = np.load('train_data_y_productBrand_one_hot_224.npy')
#y_productBrand = np.load('test_data_224_y_productBrand_one_hot.npy')

X, test_X, Y, test_Y = train_test_split(x_matrix, y_pb, 
                                        random_state=1, 
                                        train_size=0.7)

print('X and Y shape is: ', X.shape, Y.shape)
print('testX and testY shape is: ', test_X.shape, test_Y.shape)

# import pdb
# pdb.set_trace()
# acc_list = []
# cross_entropy_list = []
# acc_train = []

total_iter = 5000

class train_model(object):
    def __init__(self, sess, image_size=224, c_dim=3, cate_dim=70, batch_size=4):
        self.sess = sess
        self.image_size = image_size
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.cate_dim = cate_dim
        self.build_model()
        

    def build_model(self):
        self.img = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim],
                                  name='your_input_image')
        self.label = tf.placeholder(tf.float32, [None, self.cate_dim], name='ground_truth_label')
        
        self.hold_prob1 = tf.placeholder(tf.float32)
        
        self.output_ensemble = self.final_branch(self.googlenet(self.img, reuse=False), reuse=False)

        self.loss = tf.losses.softmax_cross_entropy(logits=self.output_ensemble['logits'],
                                                    onehot_labels=self.label,
                                                    weights=1.0,
                                                    label_smoothing=0.1)
        
        self.predict_ensemble = self.final_branch(self.googlenet(self.img, reuse=True), reuse=True)
        
        # you can consider adding label smoothing in your loss, e.g., setting label_smoothing=0.1
        # to control over fitting
        
        self.t_vars = tf.trainable_variables()

        self.t_vars1 = [var for var in self.t_vars if 'new_feats' in var.name]
        self.t_vars2 = [var for var in self.t_vars if 'new_logits' in var.name]

        self.optim_vars = self.t_vars1 + self.t_vars2

    def train(self):
        lr = 0.0001  # tune learning rate to the best performance.
        # a decay learning rate:  
        # global_step = tf.Variable(0, trainable=False)
        # lr = tf.train.exponential_decay(0.1, global_step, 2000, 0.9, staircase = False)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optim = tf.train.AdamOptimizer(lr).minimize(self.loss, var_list=self.optim_vars)
        tf.global_variables_initializer().run()

        # model download link: http: // download.tensorflow.org / models / inception_v1_2016_08_28.tar.gz

        checkpoint_path = './inception_v1.ckpt'
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()

        for var in tf.global_variables():
            dict_name = var.name
            if dict_name[:-2] in var_to_shape_map:
                print('loaded var: %s' % dict_name)
                self.sess.run(var.assign(reader.get_tensor(dict_name[:-2])))
            else:
                print('missing var %s, skipped' % dict_name)

        # import pdb
        # pdb.set_trace()
        # advice: writing training details yourself. like: _ = self.sess.run(self.loss, feed_dict={}).....

        ################################## ADDDING ##################################

        tf.summary.scalar("loss", self.loss)
        merged_summary_op = tf.summary.merge_all()
        
        saver = tf.train.Saver( max_to_keep = 1)
        checkpoint_dir = './models/googleNet_pB/'
        
        # hold_prob1 = tf.placeholder(tf.float32)
        
        shuffle_id = np.arange(len(X))
        np.random.shuffle(shuffle_id)
        shuffle_counter = 0
        for i in range (total_iter):
            print("training iter:", i)
            if shuffle_counter%5==0:
                np.random.shuffle(shuffle_id)
                shuffle_counter=0
            
            _ = sess.run(optim,feed_dict={self.img: X[shuffle_id[shuffle_counter*self.batch_size:(shuffle_counter+1)*self.batch_size]], 
                                          self.label: Y[shuffle_id[shuffle_counter*self.batch_size:(shuffle_counter+1)*self.batch_size]], 
                                          self.hold_prob1: 0.8})
            
            #if i%sample_iter==0:
            if i == total_iter - 1:
               train_loss, train_ensemble = sess.run([self.loss, self.predict_ensemble],
                                    feed_dict={self.img:X, self.label: Y, self.hold_prob1: 0.8})
               train_acc = self.predict(train_ensemble['predictions'],Y)
               test_loss, test_ensemble = sess.run([self.loss, self.predict_ensemble],
                                     feed_dict={self.img: test_X, self.label: test_Y, self.hold_prob1: 1.0})
               test_acc = self.predict(test_ensemble['predictions'], test_Y)
               print("Iteration: [%4d/%4d], train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f" % (i, total_iter, train_loss, train_acc, test_loss, test_acc))
               
               saver.save(sess, checkpoint_dir + 'googleNet_pB_model.ckpt', global_step=i+1)
               print("Checkpoint saved!")
               
            shuffle_counter = shuffle_counter + 1


        # save model
        string_time = datetime.datetime.now().strftime("%I%M%p_%B_%d_%Y")

        str_temp = 'step_1000_' + string_time + '.npy'
        save_dict = {var.name: var.eval(sess) for var in tf.global_variables()}
        np.save(str_temp, save_dict)


    def predict(self, feat, label):
    # advice: sample code for testing accuracy. You need to modify a bit to suit yourself.
        # import pdb
        # pdb.set_trace()
        total_correct_number = np.sum(np.argmax(label,axis=1) == np.argmax(feat, axis=1))
        acc = 1.0 * total_correct_number / len(label)
        
        return acc
        ################################## ADDDING ##################################
        
        
    def googlenet(self, image, reuse=False):
        assert image.get_shape().as_list()[1:] == [224, 224, 3]
        inputs = image / 127.5 - 1.0
        print(inputs)
        
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            with slim.arg_scope(inception.inception_v1_arg_scope()):
                end_points = inception_v1(inputs, is_training=False)
            return end_points

    def final_branch(self, end_points, reuse=False):
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            # [7,7] pooling size depends on your input image size. This specially targets 224x224.
            net = slim.avg_pool2d(end_points['Mixed_5c'], [7, 7], stride=1, scope='MaxPool_0a_7x7')
            net = tf.reshape(net, (-1, 1 * 1 * 1024))
            # I suggest adding another layer before final classification.
            net = slim.fully_connected(net, 1024, scope='new_feats')
            end_points['new_feats'] = net
    
            # To reduce overfitting, I suggest adopting Dropout here. Remember, Dropout
            # is used during training, but MUST NOT BE used during testing. A sample usage:
            # net = slim.dropout(net, dropout_keep_prob, scope='Dropout_0b')
            # hold_prob1 = tf.placeholder(tf.float32)
            net = tf.nn.dropout(net, keep_prob= self.hold_prob1)
            # dropout_keep_prob can be set as 0.8 or 0.5 normally during TRAINING.
            # dropout_keep_prob MUST be set as 1.0 during TESTING.
            net = slim.fully_connected(net, self.cate_dim, activation_fn=None, scope='new_logits')
            end_points['logits'] = net
            end_points['predictions'] = slim.softmax(net, scope='predictions')
    
            return end_points


if __name__ == '__main__':
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True)) as sess:
        model = train_model(sess)
        model.train()
