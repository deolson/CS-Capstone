from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell
import tensorflow as tf
class choraleModel(object):

    def __init__(self, timeNeurons, timeLayers, noteNeurons, noteLayers, dropout):
        self.timeNeurons = timeNeurons
        self.timeLayers = timeLayers
        self.noteNeurons = noteNeurons
        self.noteLayers = noteLayers
        self.dropout = dropout

        timeCell = LSTMCell(timeNeurons)
        timeStack = MultiRNNCell([timeCell]*timeLayers)

        optimizer = tf.train.AdadeltaOptimizer(1., 0.95, 1e-6)
