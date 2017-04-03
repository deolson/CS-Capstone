from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, BasicLSTMCell
import tensorflow as tf
from train_model import getModelInputs, batch_len, division_len, binary_len, batch_width
class choraleModel(object):

    def __init__(self, is_training, timeNeurons, timeLayers, noteNeurons, noteLayers, dropout):
        self.timeNeurons = timeNeurons
        self.timeLayers = timeLayers
        self.noteNeurons = noteNeurons
        self.noteLayers = noteLayers
        self.dropout = dropout

        batch = tf.placeholder(tf.float32, [batch_width, batch_len, 128, 2])
        modelInput = tf.placeholder(tf.float32, [batch_width, batch_len, 128, 80])

        def lstm_cell():
                return tf.contrib.rnn.LSTMCell(timeNeurons[0], forget_bias=0.0, state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training:
            def attn_cell():
                 return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=0.5)

        cellStack = tf.contrib.rnn.MultiRNNCell([attn_cell()*timeLayers], state_is_tuple=True)

        self._initial_state = cellStack.zero_state(batch_width, tf.float32)

        #inputs = some way to get all the inputs into this vector
        if is_training:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            inputs = tf.unstack(inputs, num=timeLayers, axis=1)
            outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=self._initial_state)

        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])
        softmax_w = tf.get_variable("softmaw_w", [size, vocab_size], dtype=tf.float32
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],[tf.reshape(input_.targets, [-1])],[tf.ones([batch_width * timeLayers], dtype=tf.float32)])

        self._cost = cost = tf.reduce_sum(loss) / batch_width
        self._final_state = state

        if not is_training:
            return

        optimizer = tf.train.AdadeltaOptimizer(1., 0.95, 1e-6)
