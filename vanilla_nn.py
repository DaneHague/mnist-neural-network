# Import dependancies
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Dump the data set into tmp
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# The batch size takes n samples of data to run through the network 
# The model then will be trained on this batch and improve accuracy
# Also when using larger data sets you have to split them up
batch_size = 128

# How many times do we want to train the network
hm_epochs = 50

# These are the entry points to the graph, x is the input, in this case images. Y is the labels
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

# Initialize xavier for the weights
init = tf.contrib.layers.xavier_initializer()

keep_prob_hl1 = tf.placeholder(tf.float32)
keep_prob_hl2 = tf.placeholder(tf.float32)


def neural_network_model(data):

	# Hidden layer 1
	h1 = tf.layers.dense(inputs=x, units=2048, activation=tf.nn.relu, kernel_initializer=init)

	h_fc1_drop = tf.nn.dropout(h1, keep_prob_hl1)

	# Hidden layer 2
	h2 = tf.layers.dense(inputs=h_fc1_drop, units=2048, activation=tf.nn.relu)

	h2_fc2_drop = tf.nn.dropout(h2, keep_prob_hl2)

	# Output layer
	y_pred = tf.layers.dense(inputs=h2_fc2_drop, units=10)

	return y_pred

def train_neural_network(x):
	prediction = neural_network_model(x)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

	optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3, beta1 = 0.9, beta2 = 0.999).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y, keep_prob_hl1: 0.5, keep_prob_hl2: 0.5})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels, keep_prob_hl1: 1.0, keep_prob_hl2: 1.0}))

		saver = tf.train.Saver()
		saver.save(sess, 'model/model.ckpt')

train_neural_network(x)