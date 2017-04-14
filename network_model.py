from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, BasicLSTMCell
import tensorflow as tf
from train_model import getModelInputs, batch_len, division_len, binary_len, batch_width
import numpy as numpy

class choraleModel(object):

    def __init__(self, is_training, timeNeurons, timeLayers, noteNeurons, noteLayers, dropout):

        iterations = 1;

        with tf.Session() as sess:
            # inputs to the model, batch is a submatrix of the statematrix that is fitted to our batchs and modelInput is the note vectors
            batch = tf.placeholder(tf.float32, [batch_width, batch_len, 128, 2]) # (batch, time, notes, 2)
            modelInput = tf.placeholder(tf.float32, [batch_len-1, batch_width*128, 80]) # subtract the last layer we don't need its output (time-1, batch*notes, vector len)
            # Tensorflow needs 3d array so modelInput [[vector],[vector],[vector]->128*batch],[[vector],[vector],[vector]->128*batch]
            #                                          ------------------------------------time--------------------------------------

            with tf.variable_scope("time"):
                # LSTMCell make a whole layer, in this case with 300 neurons, and timeStack is both of the layers
                timeStack = tf.contrib.rnn.MultiRNNCell([LSTMCell(timeNeurons[0],state_is_tuple=True) for _ in range(timeLayers)], state_is_tuple=True)
                #we pass our layers to dynamic_rnn and it loops through our modelInput and trains the weights
                timeOutputs, _ = tf.nn.dynamic_rnn(timeStack, modelInput, dtype=tf.float32, time_major=True)

            # timeOutputs is the outputs at the last timestep, current of shape (time-1, batch*notes, vector len)
            # In order to now loop throguh the notes in the note layer we need to get (note,time*batch,hiddens)
            # first reshape to (time,batch,notes,hiddens) -> (notes, batch, time, hiddens) -> (note, time*batch,hiddens)
            timeFin = tf.reshape(timeOutputs, [batch_len-1,batch_width,128,300])
            timeFin = tf.transpose(timeFin, [2,1,0,3])
            timeFin = tf.reshape(timeFin, [128,batch_width*(batch_len-1),300])

            # creates a numpy array that will match our input into the note layer
            # (1,batch*time,2)
            tmp = numpy.zeros(shape=(1,batch_width*(batch_len-1),2))
            tmp = tf.stack(tmp)
            tmp = tf.cast(tmp, tf.float32)

            # next we take the actual notes that were played in the form (batch, time, notes, 2) and ignore the first time: 0
            # we ignore time: 0 because we are passing in the notes that where actually played at the next time step
            # we the transpose that to (notes,batch,time,2) -> (127, batch*time,2)
            # then pad the first layer with 0s from tmp to get (notes,time*batch,2)
            # now our actual_notes match the dimensions of our timeFin dimensions
            actual_note = tf.transpose(batch[:,1:,0:-1,:], [2,0,1,3])
            actual_note = tf.reshape(actual_note, [127,batch_width*(batch_len-1),2])
            actual_note = tf.concat([tmp,actual_note],0)

            # we take the timeFin and smoosh it with the actual notes played along the 2 axis
            # this means that the actual notes played are added with the 300 hiddens we are passing in
            # (notes,batch*time,hiddens+batchs)
            noteLayer_input = tf.concat([timeFin,actual_note],2)

            with tf.variable_scope("note"):
                noteStack = tf.contrib.rnn.MultiRNNCell([LSTMCell(noteNeurons[i],state_is_tuple=True) for i in range(noteLayers)], state_is_tuple=True)
                noteOutputs, _ = tf.nn.dynamic_rnn(noteStack, noteLayer_input, dtype=tf.float32, time_major=False)

            # noteFine = tf.reshape(noteOutputs, [128,batch_width,(batch_len-1),2])
            # noteFine= tf.transpose(noteFin, [1,2,0,3])



            # print("-----------")
            # print(actual_note)
            # print("-----------")

            # cross_entropy = tf.reduce_mean(-tf.reduce_sum(modelInput * tf.log(y), reduction_indices=[1]))
            # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=[1.0], logits=[1.0]))
            # train_step = 0

            sess.run(tf.global_variables_initializer())
            for i in range(iterations):
                inputBatch, inputModelInput = getModelInputs()


                # train_step.run([optimizer,cross_entropy],feed_dict={batch: inputBatch, modelInput:inputModelInput})
                # train_step.run(feed_dict={batch: inputBatch, modelInput:inputModelInput})
                # print(sess.run([noteOutputs],feed_dict={batch: inputBatch, modelInput:inputModelInput}))
                print("==================================")
                print("==================================")
                print("==================================")
                print(sess.run([noteOutputs],feed_dict={batch: inputBatch, modelInput:inputModelInput})[0].shape)
                # print(val.eval())

            sess.close()

    def timeFun():
        pass
