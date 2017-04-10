from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, BasicLSTMCell
import tensorflow as tf
from train_model import getModelInputs, batch_len, division_len, binary_len, batch_width

class choraleModel(object):

    def __init__(self, is_training, timeNeurons, timeLayers, noteNeurons, noteLayers, dropout):
        iterations = 1;
        with tf.Session() as sess:
            batch = tf.placeholder(tf.float32, [batch_width, batch_len, 128, 2])
            modelInput = tf.placeholder(tf.float32, [batch_len-1, batch_width*128, 80])

            timeStack = tf.contrib.rnn.MultiRNNCell([LSTMCell(timeNeurons[0],state_is_tuple=True) for _ in range(timeLayers)], state_is_tuple=True)
            val, hidden_state = tf.nn.dynamic_rnn(timeStack, modelInput, dtype=tf.float32, time_major=True)


            # print("-----------")
            # print(val.eval())
            # print("-----------")

            # W = tf.Variable(tf.zeros([batch_len-1, 80, 80]))
            # y = tf.nn.softmax(tf.matmul(modelInput,W))
            # cross_entropy = tf.reduce_mean(-tf.reduce_sum(modelInput * tf.log(y), reduction_indices=[1]))


            # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=[1.0], logits=[1.0]))
            # train_step = 0


            sess.run(tf.global_variables_initializer())
            for i in range(iterations):
                inputBatch, inputModelInput = getModelInputs()
                # train_step.run([optimizer,cross_entropy],feed_dict={batch: inputBatch, modelInput:inputModelInput})
                # train_step.run(feed_dict={batch: inputBatch, modelInput:inputModelInput})
                print(sess.run([hidden_state],feed_dict={batch: inputBatch, modelInput:inputModelInput}))
                # print(val.eval())

            sess.close()












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
