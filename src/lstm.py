from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import GRUCell
import numpy as np
import time
import os

class RNN():
    def __init__(self, num_classes, batch_size=64, seq_length=600, embed_size=100, cell_type='LSTM',
                 rnn_size=128, num_layers=2, learning_rate=0.001, train_keep_prob=0.5, sampling=False):
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
        #self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        
        self.inputs_layer()
        self.rnn_layer()
        self.outputs_layer()
        self.my_loss()
        self.val_acc()
        #self.my_optimizer()
        self.saver = tf.train.Saver()
        
    def inputs_layer(self):

        self.inputs = tf.placeholder(tf.float32, shape=(self.batch_size, self.seq_length, self.embed_size), name='inputs')
        self.targets = tf.placeholder(tf.int32, shape=(self.batch_size), name='targets')
        
    def rnn_layer(self):
        
        if (self.cell_type == 'LSTM'):
            cell = LSTMCell(self.rnn_size)
        else:
            cell = GRUCell(self.rnn_size)
            
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

    def train(self, batches, X_val, y_val):
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                counter = 0

                # Initialize the states of the RNN
                new_state = sess.run(self.initial_state)

                # Train network
                print("Initializing training")
                for X_batch, y_batch in batches:
                    counter += 1
                    start = time.time()
                    feed = {self.inputs: X_batch,
                            self.targets: y_batch,
                            #self.keep_prob: self.train_keep_prob,
                            self.initial_state: new_state}
                    batch_loss, _, new_state = sess.run([self.loss, self.optimizer,
                                                         self.final_state],
                                                         feed_dict=feed)
                    end = time.time()
                    if counter % 1 == 0:
                        print('step: {} '.format(counter),
                              'loss: {:.4f} '.format(batch_loss),
                              '{:.4f} sec/batch'.format((end-start)))
                    if counter % 5 == 0:
                        feed = {self.inputs: X_val,
                                self.targets: y_val,
                                self.initial_state: new_state}
                        val_acc, preds, targs = sess.run([self.accuracy,self.preds, self.targets], feed_dict = feed)
                        print('step: {} '.format(counter),
                              'validation accuracy {} '.format(val_acc))
                        print(preds)
                        print(targs)

                    '''
                    if (counter % save_every_n == 0):
                        print("Model Saved")
                        self.saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, self.rnn_size))
                        
                    if counter >= max_count:
                        break
                    '''

                #self.saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, self.rnn_size))
    def predict(self, X_test):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            # Need to save best training model above and load it here
            new_state = sess.run(self.initial_state)

            # Run predictions
            print("Running network predictions")
            feed = {self.inputs: X_test,
                    #self.keep_prob: self.train_keep_prob,
                    self.initial_state: new_state}
            predictions = sess.run([self.preds], feed_dict=feed)

        return predictions

    # Optimizer below is from HW3 - includes gradient clip
    '''
    def my_optimizer(self):
        
        build our optimizer
        Unlike previous worries of gradient vanishing problem,
        for some structures of rnn cells, the calculation of hidden layers' weights 
        may lead to an "exploding gradient" effect where the value keeps growing.
        To mitigate this, we use the gradient clipping trick. Whenever the gradients are updated, 
        they are "clipped" to some reasonable range (like -5 to 5) so they will never get out of this range.
        parameters we will use:
        self.loss, self.grad_clip, self.learning_rate
        we have to define:
        self.optimizer for later use
        
        # using clipping gradients
        #######################################################
        # TODO: implement your optimizer with gradient clipping
        #######################################################
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # Gradient clipping - optimizer.minimize(loss) both calculates gradients and applies them
        # Here, we would like to calculate, then manipulate, and then apply
        
        # https: // stackoverflow.com / questions / 36498127 / how - to - apply - gradient - clipping - in -tensorflow
        gvs = optimizer.compute_gradients(self.loss)
        #https: // github.com / jazzsaxmafia / Inpainting / issues / 6
        capped_gvs = map(lambda gv: gv if gv[0] is None \
                         else [tf.clip_by_value(gv[0], -self.grad_clip, self.grad_clip), gv[1]], gvs)
        self.optimizer = optimizer.apply_gradients(capped_gvs) 
    '''