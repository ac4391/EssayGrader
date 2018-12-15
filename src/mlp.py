import tensorflow as tf
import numpy as np

class MLP(object):

    def __init__(self, input_dim=200, hidden_dims=[200, 200], num_classes=12, weight_scale=1e-2,
                 l2_reg=1e-2, keep_prob=0.8, regression=False, use_bn=None):
        """
        Inputs:
        :param input_dim: size of the input layer
        :param hidden_dims: size of the hidden layers (as a list)
        :param num_classes: number of output classes for classification
        :param weight_scale: (float) for layer weight initialization
        :param l2_reg: (float) L2 regularization
        :param reg: boolean indicating whether regression is active. Otherwise
                    classification.
        :param use_bn: (bool) decide whether to use batch normalization or not
        :param dropout_config: (dict) configuration for dropout
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.reg = regression # Use regression if true, classification if false

        # Regression will have an output size of 1
        if self.reg:
            self.output_dim = 1
        else:
            self.output_dim = num_classes

        self.l2_reg = l2_reg
        self.X = tf.placeholder(tf.float32, shape=(None, self.input_dim))
        if self.reg:
            self.y = tf.placeholder(tf.float32, shape=(None,))
        else:
            self.y = tf.placeholder(tf.int32, shape=(None,))
        self.keep_prob = keep_prob

        # Define input layer weights and biases
        self.W = [tf.Variable(weight_scale * np.random.rand(self.input_dim, self.hidden_dims[0]).astype('float32'))]
        self.B = [tf.Variable(np.zeros((self.hidden_dims[0],)).astype('float32'))]

        # Add weights and biases for each hidden layer
        for idx in range(len(self.hidden_dims)-1):
            self.W.append(tf.Variable(weight_scale * np.random.rand(self.hidden_dims[idx], self.hidden_dims[idx+1]).astype('float32')))
            self.B.append(tf.Variable(np.zeros((self.hidden_dims[idx+1],)).astype('float32')))

        # Add weights and biases for output layer
        self.W.append(tf.Variable(weight_scale * np.random.rand(self.hidden_dims[-1], self.output_dim).astype('float32')))
        self.B.append(tf.Variable(np.zeros((self.output_dim,)).astype('float32')))

        # Create input layer, hidden layers, and output layer
        self.layers = [tf.nn.relu(tf.matmul(self.X, self.W[0]) + self.B[0])]
        for idx in range(len(hidden_dims)-1):
            curr_layer = tf.nn.relu(tf.matmul(self.layers[idx], self.W[idx+1]) + self.B[idx+1])
            curr_layer = tf.nn.dropout(curr_layer, keep_prob=self.keep_prob)
            self.layers.append(curr_layer)
        self.outputs = tf.matmul(self.layers[-1], self.W[-1]) + self.B[-1]

        self.loss()
        self.accuracy()

    def accuracy(self):
        # Prediction is different for regression and classification
        if self.reg:
            self.preds = tf.squeeze(tf.cast(tf.round(self.outputs), tf.int32))
        else:
            # Prediction for classification is the class with the highest probability
            self.preds = tf.argmax(self.outputs, axis=1, output_type=tf.int32)

        # Check for equality of prediction and label. Save accuracy over batch
        correct_preds = tf.equal(tf.cast(self.y, tf.int32), self.preds)
        #correct_preds = tf.equal(tf.cast(self.y, tf.int32), tf.cast(tf.round(self.preds), tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    def loss(self):
        if self.reg==True:
            cost = tf.reduce_mean(tf.square(self.outputs-self.y))
        else:
            cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.outputs, labels=tf.one_hot(self.y, self.output_dim))

        # Incorporate L2 regularization
        L2_loss = sum([tf.nn.l2_loss(w) for w in self.W])
        self.loss = tf.reduce_mean(cost) + self.l2_reg*L2_loss

    def optimizer_def(self, lr):
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = self.optimizer.minimize(self.loss)

    def train(self, gen, X_val, y_val, n_epochs, n_batches, lr, save_every_n, model_name):
        self.optimizer_def(lr)

        best_acc = 0
        init = tf.global_variables_initializer()
        train_loss_hist = {}
        val_loss_hist = {}
        with tf.Session() as sess:
            self.saver = tf.train.Saver()
            sess.run(init)
            for e in range(1,n_epochs+1):
                print('\n')
                print('-' * 10, 'Training epoch: {}'.format(e), '-' * 10, end='')
                for itr in range(1,n_batches+1):
                    batch_X, batch_y = next(gen)
                    loss, _ = sess.run([self.loss,self.train_op], feed_dict={self.X:batch_X, self.y:batch_y})

                    if itr%save_every_n ==0:
                        train_loss_hist[(e,itr)] = loss
                        val_loss = sess.run([self.loss], feed_dict={self.X: X_val, self.y: y_val})
                        val_loss_hist[(e,itr)] = val_loss

                    if e%1==0 and itr==1:
                        preds, vals, val_acc = sess.run([self.preds, self.y, self.accuracy],\
                                                        feed_dict={self.X: X_val,
                                                                   self.y: y_val})
                        print('Epoch {}, Batch {} -- Loss: {:0.3f} Validation accuracy: {:0.3f}'.format(e,itr,loss,val_acc))
                        vals = vals.astype('int32')
                        print("Sample Grade Predictions: ")
                        print("Preds:  ", *preds[10:30], sep=' ')
                        print("Actual: ", *vals[10:30], sep=' ')

                        if val_acc > best_acc:
                            best_acc = val_acc
                            print('Best validation accuracy! - Saving Model')
                            self.saver.save(sess, 'model/'+model_name)

            print('Best validation accuracy over the training period was: {}%'.format(best_acc))

        return train_loss_hist, val_loss_hist

    def predict(self, checkpoint, X_test):
        with tf.Session() as sess:
            # Restore a saved model and run prediction
            self.saver.restore(sess, checkpoint)
            preds = sess.run([self.preds], feed_dict={self.X: X_test})

        return preds[0]
    