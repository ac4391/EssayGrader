from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import GRUCell
import numpy as np
import time
import os

class RNN():
    def __init__(self, num_classes, batch_size=64, seq_length=600, embed_size=100, cell_type='LSTM',
                 rnn_size=128, num_layers=2, learning_rate=0.001, train_keep_prob=0.8, sampling=False):
        '''
        Initialize the input parameter to define the network
        inputs:
        :param num_classes: (int) the vocabulary size of your input data
        :param batch_size: (int) number of sequences in one batch
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
        self.seq_length = seq_length
        self.embed_size = embed_size
        self.cell_type = cell_type
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.train_keep_prob = train_keep_prob
        
        self.inputs_layer()
        self.rnn_layer()
        self.outputs_layer()
        self.my_loss()
        self.val_acc()
        self.saver = tf.train.Saver()
        
    def inputs_layer(self):

        self.inputs = tf.placeholder(tf.float32, shape=(None, self.seq_length, self.embed_size), name='inputs')
        self.targets = tf.placeholder(tf.int32, shape=(None), name='targets')
        
    def rnn_layer(self):
        
        if self.cell_type == 'LSTM':
            cell = LSTMCell(self.rnn_size)
        else:
            cell = GRUCell(self.rnn_size)

        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.train_keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell], state_is_tuple=True)

        # Initial state
        self.initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.inputs,\
                                                               initial_state=self.initial_state, dtype=tf.float32)
        
    def outputs_layer(self):
        seq_output = self.rnn_outputs[:,-1,:] # We only want the output from the last iteration
        x = tf.reshape(seq_output, [-1, self.rnn_size])
        
        # define softmax layer variables:
        with tf.variable_scope('softmax'):
            # Not sure why we use this normal distribution for initialization
            softmax_w = tf.Variable(tf.truncated_normal([self.rnn_size, self.num_classes], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(self.num_classes))
        
        # calculate logits
        self.logits = tf.matmul(x, softmax_w) + softmax_b
        
        # softmax generate probability predictions
        self.prob_pred = tf.nn.softmax(self.logits, name='predictions')
        
    def my_loss(self):
        y_one_hot = tf.one_hot(self.targets, self.num_classes)
        y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
        
        # Softmax cross entropy loss
        # Shoule we use logits as self.logits or self.prob_pred (after softmax) ?
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y_reshaped)
        self.loss = tf.reduce_mean(loss)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def val_acc(self):
        # Calculate prediction as the maximum probability output from the output layer
        self.preds = tf.argmax(self.prob_pred, axis=1, output_type=tf.int32)

        # Test for equality of target and prediction. Validation accuracy is the
        # mean of the resulting array, representing the proportion of correct preds
        correct_prediction = tf.equal(self.targets, self.preds)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, gen, X_val, y_val, n_epochs, n_batches, save_every_n, model_name):

            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)

                # Initialize the states of the RNN
                new_state = sess.run(self.initial_state)
                best_acc = 0.0
                train_loss_hist = {}
                val_loss_hist = {}

                # Train network
                for e in range(1, n_epochs + 1):
                    print('\n')
                    print('-'*10, 'Training epoch: {}'.format(e), '-'*10)
                    for itr in range(1, n_batches + 1):
                        batch_X, batch_y = next(gen)
                        start = time.time()
                        feed = {self.inputs: batch_X,
                                self.targets: batch_y,
                                self.initial_state: new_state}
                        batch_loss, _, new_state = sess.run([self.loss, self.optimizer,
                                                             self.final_state],
                                                             feed_dict=feed)
                        end = time.time()

                        # Save training and validation loss for plotting
                        if itr%save_every_n == 0:
                            train_loss_hist[(e,itr)] = batch_loss
                            val_loss = sess.run([self.loss], feed_dict={self.inputs: X_val,
                                                                        self.targets: y_val,
                                                                        self.initial_state: new_state})
                            val_loss_hist[(e,itr)] = val_loss

                        if e%1==0 and itr%5==0:
                            feed = {self.inputs: X_val,
                                    self.targets: y_val,
                                    self.initial_state: new_state}
                            val_acc, preds, targs = sess.run([self.accuracy,self.preds, self.targets], feed_dict = feed)
                            print('Epoch {}, step {}'.format(e, itr),
                                  'loss: {:.4f} '.format(batch_loss),
                                  'validation accuracy: {} '.format(val_acc),
                                  '{:.4f} sec/batch'.format((end-start)))

                            # Early stopping: save best model
                            if val_acc > best_acc:
                                best_acc = val_acc
                                print('Best validation accuracy! - Saving Model')
                                self.saver.save(sess, 'model/'+model_name)

                            # Ouput a subset of predictions to the user
                            print("Sample Grade Predictions: ")
                            print("Preds:  ", *preds[:20], sep=' ')
                            print("Actual: ", *targs[:20], sep=' ')

                print('Best validation accuracy over the training period was: {}%'.format(best_acc))

            return train_loss_hist, val_loss_hist

    def predict(self, checkpoint, X_test):
        with tf.Session() as sess:
            # Load training model from checkpoint for predictions
            self.saver.restore(sess, checkpoint)

            # Run predictions
            print("Running network predictions")
            feed = {self.inputs: X_test}
            predictions = sess.run([self.preds], feed_dict=feed)

        return predictions

