from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell

class choraleModel(object):

    def __init__(self, timeNeurons, timeLayers, noteNeurons, noteLayers, dropout):
        self.timeNeurons = timeNeurons
        self.timeLayers = timeLayers
        self.noteNeurons = noteNeurons
        self.noteLayers = noteLayers
        self.dropout = dropout

        timeCell = LSTMCell(timeNeurons)
        timeStack = MultiRNNCell([timeCell]*timeLayers)
