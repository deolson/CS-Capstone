from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, BasicLSTMCell
import tensorflow as tf
from train_model import getModelInputs
class choraleModel(object):

    def __init__(self, timeNeurons, timeLayers, noteNeurons, noteLayers, dropout):
        self.timeNeurons = timeNeurons
        self.timeLayers = timeLayers
        self.noteNeurons = noteNeurons
        self.noteLayers = noteLayers
        self.dropout = dropout

        with tf.Session() as sess:
            data = tf.placeholder(tf.float32,[None,80,1])

            timeCell = tf.contrib.rnn.LSTMCell(timeNeurons[0], state_is_tuple=True)
            cellStack = tf.contrib.rnn.MultiRNNCell([timeCell]*timeLayers)

            val, state = tf.nn.dynamic_rnn(timeCell, data, dtype=tf.float32)

            #for i  in range(num_steps):
                #output, state = lstm


            # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
            optimizer = tf.train.AdadeltaOptimizer(1., 0.95, 1e-6)
