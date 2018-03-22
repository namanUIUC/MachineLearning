import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

class ConvNet:
    def __init__(self, sess):
        self.sess = sess
        
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        x_image = tf.reshape(x, [-1, 28, 28, 1])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        
        self.x = x
        self.y_ = y_
        self.y_conv = y_conv
        self.keep_prob = keep_prob

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        self.train_step = train_step
        self.accuracy = accuracy
        self.correct_prediction = correct_prediction
        
        sess.run(tf.global_variables_initializer())
    
    def train(self, data, n_iter, logging=100, keep_prob_train = 0.5):
        for i in range(n_iter):
            batch = data.train.next_batch(50)
            if logging > 0 and i % logging == 0:
                train_accuracy = self.accuracy.eval(feed_dict={
                    self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
            self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: keep_prob_train})

        print("test accuracy %g" % self.accuracy.eval(feed_dict={
            self.x: data.test.images, self.y_: data.test.labels, self.keep_prob: 1.0}))
    
    def test_accuracy(self, x, y):
        return self.correct_prediction.eval(feed_dict={
            self.x: x,
            self.y_: y,
            self.keep_prob: 1.0
        })
        
    def get_input(self):
        return self.x
    
    def get_raw_output(self):
        return self.y_conv
    
    def get_labels(self):
        return self.y_
    


    