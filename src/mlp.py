import tensorflow as tf
import numpy as np
from src.utils import normalize_predictions

class MLP(object):

    def __init__(self, input_dim=60000, hidden_dims=[200, 200], num_classes=12, weight_scale=1e-2,
                 l2_reg=1e-2, reg=False, use_bn=None, dropout_config=None):
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
        self.output_dim = num_classes
        self.l2_reg = l2_reg
        self.reg = reg # Use regression if true, classification if false

        self.X = tf.placeholder(tf.float32, shape=(None, self.input_dim))
        if self.reg:
            self.y = tf.placeholder(tf.float32, shape=(None,))
        else:
            self.y = tf.placeholder(tf.int32, shape=(None,))


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
            self.layers.append(tf.nn.relu(tf.matmul(self.layers[idx], self.W[idx+1]) + self.B[idx+1]))
        self.outputs = tf.matmul(self.layers[-1], self.W[-1]) + self.B[-1]

        # Outputs for regression are single values. Outputs for classification
        # are array of probabilities for each class
        if self.reg:
            self.outputs = tf.argmax(self.outputs, axis=1, output_type=tf.int32)

        self.loss()
        self.accuracy()

    def accuracy(self):
        if self.reg:
            self.preds = tf.cast(tf.round(self.outputs), tf.float32)
        else:
            self.preds = tf.argmax(self.outputs, axis=1, output_type=tf.int32)

        # Check for equality of prediction and label. Save accuracy over batch
        correct_preds = tf.equal(tf.cast(self.y, tf.int32), self.preds)
        self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))


    def loss(self):
        # Should this be sum of the squares rather than mean?
        if self.reg==True:
            cost = tf.reduce_mean(tf.square(tf.cast(self.outputs, tf.float32)-self.y))
        else:
            cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.outputs, labels=tf.one_hot(self.y, self.output_dim))

        # Incorporate L2 regularization
        L2_loss = sum([tf.nn.l2_loss(w) for w in self.W])
        self.loss = tf.reduce_mean(cost) + self.l2_reg*L2_loss

    def optimizer_def(self, lr):
        self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)


    def train(self, gen, X_val, y_val, s_val, n_epochs, lr):
        self.optimizer_def(lr)

        self.train_step = self.optimizer
        best_acc = 0
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(init)
            counter = 0
            for batch_X, batch_y, batch_s in gen:
                counter += 1
                loss, _ = sess.run([self.loss,self.train_step], feed_dict={self.X: batch_X, self.y: batch_y})
                if counter % 100 == 0:
                    preds, vals, val_acc = sess.run([self.preds, self.y, self.accuracy], feed_dict={self.X: X_val, self.y: y_val})
                    print("loss for counter {} is {}".format(counter, loss))
                    print('counter {}: valid acc = {}'.format(counter, val_acc))
                    preds = normalize_predictions(preds, s_val)
                    vals = normalize_predictions(vals, s_val)
                    print(preds[:20])
                    print(vals[:20])
                    #if val_acc > best_acc:
                     #   print('Best validation accuracy! iteration:{} accuracy: {}%'.format(counter, val_acc))
                      #  best_acc = val_acc
                       # self.saver.save(sess, 'model/{}'.format(counter))
                    if counter > n_epochs * 1000:
                        break
    
    def predict(self, checkpoint, X_test, X_set):
        
        self.session = tf.Session()
        with self.session as sess:
            predictions = tf.zeros(shape = X_test.shape)
            self.saver.restore(sess, checkpoint)
            for i, x in enumerate(X_test):
                preds = sess.run(self.preds, feed_dict={self.X: x})
                norm_pred = normalize_predictions(preds[0], X_set[i])
                predictions[i] = norm_pred
        return predictions
    