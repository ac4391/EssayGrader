import tensorflow as tf
import numpy as np
from src.utils import normalize_predictions

class MLP(object):

    def __init__(self, input_dim=60000, hidden_dims=[200, 200], num_classes=12, weight_scale=1e-2,
                 l2_reg=1e-2, use_bn=None, dropout_config=None):
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
        self.W2 = tf.Variable(1e-2 * np.random.rand(self.hidden_dims[0], self.output_dim).astype('float32'))
        self.b2 = tf.Variable(np.zeros((self.output_dim,)).astype('float32'))

        h1_tf = tf.nn.relu(tf.matmul(self.X, self.W1) + self.b1)
        h2_tf = tf.matmul(h1_tf, self.W2) + self.b2


        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=h2_tf, labels=tf.one_hot(self.y, num_classes))
        L2_loss = tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2) + tf.nn.l2_loss(self.W2)
        self.loss = tf.reduce_mean(cross_entropy) + l2_reg*L2_loss

        self.preds = tf.argmax(h2_tf,1)
        correct_prediction = tf.equal(self.y, self.preds)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.saver = tf.train.Saver()

    def train(self, gen, X_val, y_val, n_epochs, lr):

        self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)
        best_acc = 0
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(init)
            counter = 0
            for batch_X, batch_y in gen:
                counter += 1
                loss, _ = sess.run([self.loss,self.train_step], feed_dict={self.X: batch_X, self.y: batch_y})
                if counter % 7000 == 0:
                    preds,vals, val_acc = sess.run([self.preds,self.y, self.accuracy], feed_dict={self.X: X_val, self.y: y_val})
                    print("loss for counter {} is {}".format(counter, loss))
                    print('counter {}: valid acc = {}'.format(counter, val_acc))
                    print(preds[:20])
                    print(vals[:20])
                    if val_acc > best_acc:
                        print('Best validation accuracy! iteration:{} accuracy: {}%'.format(counter, val_acc))
                        best_acc = val_acc
                        self.saver.save(sess, 'model/{}'.format(counter))
                    if counter > n_epochs * 7000:
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
    