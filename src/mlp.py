import tensorflow as tf
import numpy as np


class MLP(object):

    def __init__(self, input_dim=60000, hidden_dims=[200, 200], num_classes=12, weight_scale=1e-2,
                 l2_reg=0.0, use_bn=None, dropout_config=None):
        """
        Inputs:
        :param weight_scale: (float) for layer weight initialization
        :param l2_reg: (float) L2 regularization
        :param use_bn: (bool) decide whether to use batch normalization or not
        :param dropout_config: (dict) configuration for dropout
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = num_classes

        self.X = tf.placeholder(tf.float32, shape=(None, self.input_dim))
        self.y = tf.placeholder(tf.int64, shape=(None,))

        self.W1 = tf.Variable(1e-2 * np.random.rand(self.input_dim, self.hidden_dims[0]).astype('float32'))
        self.b1 = tf.Variable(np.zeros((self.hidden_dims[0],)).astype('float32'))
        self.W2 = tf.Variable(1e-2 * np.random.rand(self.hidden_dims[0], self.hidden_dims[1]).astype('float32'))
        self.b2 = tf.Variable(np.zeros((self.hidden_dims[1],)).astype('float32'))
        self.W3 = tf.Variable(1e-2 * np.random.rand(self.hidden_dims[1], self.output_dim).astype('float32'))
        self.b3 = tf.Variable(np.zeros((self.output_dim,)).astype('float32'))

        h1_tf = tf.nn.relu(tf.matmul(self.X, self.W1) + self.b1)
        h2_tf = tf.nn.relu(tf.matmul(h1_tf, self.W2) + self.b2)
        h3_tf = tf.matmul(h2_tf, self.W3) + self.b3

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=h3_tf, labels=tf.one_hot(self.y, num_classes))
        L2_loss = tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2) + tf.nn.l2_loss(self.W3)
        self.loss = tf.reduce_mean(cross_entropy) * L2_loss

        correct_prediction = tf.equal(self.y, tf.argmax(h3_tf, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, gen, X_val, y_val, n_epochs, lr):

        train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            counter = 0
            for batch_X, batch_y in gen:
                counter += 1
                loss, _ = sess.run([self.loss,train_step], feed_dict={self.X: batch_X, self.y: batch_y})
                if counter % 5 ==0:
                    val_acc = sess.run([self.accuracy], feed_dict={self.X: X_val, self.y: y_val})
                    print("loss for counter {} is {}".format(counter, loss))
                    print('counter {}: valid acc = {}'.format(counter, val_acc))
