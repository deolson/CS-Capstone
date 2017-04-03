from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, BasicLSTMCell
import tensorflow as tf
from train_model import getModelInputs, batch_len, division_len, binary_len, batch_width
class choraleModel(object):

    def __init__(self, is_training, timeNeurons, timeLayers, noteNeurons, noteLayers, dropout):
        iterations = 10;

        batch = tf.placeholder(tf.float32, [batch_width, batch_len, 128, 2])
        modelInput = tf.placeholder(tf.float32, [batch_len-1, batch_width*128, 80])

        timeStack = tf.contrib.rnn.MultiRNNCell([LSTMCell(timeNeurons[0],state_is_tuple=True) for _ in range(timeLayers)], state_is_tuple=True)
        val, hidden_state = tf.nn.dynamic_rnn(timeStack, modelInput, dtype=tf.float32, time_major=True)



        # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        # train_step = optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(iterations):
                inputBatch, inputModelInput = getModelInputs()
                # train_step.run([optimizer,cross_entropy],feed_dict={batch: inputBatch, modelInput:inputModelInput})













        # def lstm_cell():
        #         return tf.contrib.rnn.LSTMCell(timeNeurons[0], forget_bias=0.0, state_is_tuple=True)

        # attn_cell = lstm_cell
        # if is_training:
        #     def attn_cell():
        #          return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=0.5)
        #
        # cellStack = tf.contrib.rnn.MultiRNNCell([attn_cell()*timeLayers], state_is_tuple=True)
        #
        # self._initial_state = cellStack.zero_state(batch_width, tf.float32)
        #
        # #inputs = some way to get all the inputs into this vector
        # if is_training:
        #     inputs = tf.nn.dropout(inputs, config.keep_prob)
        #
        # outputs = []
        # state = self._initial_state
        # with tf.variable_scope("RNN"):
        #     inputs = tf.unstack(inputs, num=timeLayers, axis=1)
        #     outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=self._initial_state)

        # if not is_training:
        #     return
