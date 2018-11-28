from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import GRUCell
import numpy as np
import time
import os

class LSTM_RNN():
    def __init__(self, num_classes, batch_size=64, num_steps=500, cell_type='LSTM',
                 rnn_size=128, num_layers=2, learning_rate=0.001, train_keep_prob=0.5, sampling=False):
        '''
        Initialize the input parameter to define the network
        inputs:
        :param num_classes: (int) the vocabulary size of your input data
        :param batch_size: (int) number of sequences in one batch
        :param num_steps: (int) length of each sequence in one batch
        :param cell_type: your rnn cell type, 'LSTM' or 'GRU'
        :param rnn_size: (int) number of units in one rnn layer
        :param num_layers: (int) number of rnn layers
        :param learning_rate: (float)
        :param train_keep_prob: (float) dropout probability for rnn cell training
        :param sampling: (boolean) whether train mode or sample mode
        '''

        tf.reset_default_graph()
        
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.cell_type = cell_type
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        
        self.inputs_layer()
        self.rnn_layer()
        self.outputs_layer()
        self.my_loss()
        self.my_optimizer()
        self.saver = tf.train.Saver()
        
    def inputs_layer(self):

        self.inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps), name='inputs')
        self.targets = tf.placeholder(tf.int32, shape=(self.batch_size, 1), name='targets')
        
    def rnn_layer(self):
        
        if (cell_type == 'LSTM'):
            cell = LSTMCell(self.rnn_size)
        else:
            cell = GRUCell(self.rnn_size)
            
        cell = tf.nn.rnn_cell.MultiRNNCell([cell], state_is_tuple=True)
        self.initial_state = cell.zero_state(self.batch_size, dtype = tf.float32)
        
        
        self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.inputs, initial_state = self.initial_state, dtype=tf.float32)
        
    def outputs_layer(self):

        seq_output = self.rnn_outputs #Perhaps we need to transpose this
        x = tf.reshape(seq_output, [-1, self.rnn_size])
        
        # define softmax layer variables:
        with tf.variable_scope('softmax'):
            softmax_w = tf.Variable(tf.truncated_normal([self.rnn_size, self.num_classes], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(self.num_classes))
        
        # calculate logits
        self.logits = tf.matmul(x, softmax_w) + softmax_b
        
        # softmax generate probability predictions
        self.prob_pred = tf.nn.softmax(self.logits, name='predictions')
        
    def my_loss(self):
    
        y_reshaped = tf.reshape(self.targets, self.logits.get_shape())
        
        # Softmax cross entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y_reshaped)
        self.loss = tf.reduce_mean(loss)     
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        