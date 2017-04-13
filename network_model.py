from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, BasicLSTMCell
import tensorflow as tf
from train_model import getModelInputs, batch_len, division_len, binary_len, batch_width
import numpy as numpy

class choraleModel(object):

    def __init__(self, is_training, timeNeurons, timeLayers, noteNeurons, noteLayers, dropout):
        iterations = 1;
        with tf.Session() as sess:
            batch = tf.placeholder(tf.float32, [batch_width, batch_len, 128, 2])

            modelInput = tf.placeholder(tf.float32, [batch_len-1, batch_width*128, 80])

            with tf.variable_scope("time"):
                timeStack = tf.contrib.rnn.MultiRNNCell([LSTMCell(timeNeurons[0],state_is_tuple=True) for _ in range(timeLayers)], state_is_tuple=True)
                timeOutputs, _ = tf.nn.dynamic_rnn(timeStack, modelInput, dtype=tf.float32, time_major=True)

            # hiddenNum = timeOutputs.shape[2]
            # noteStack = tf.contrib.rnn.MultiRNNCell([LSTMCell(noteNeurons[i],state_is_tuple=True) for i in range(noteLayers)], state_is_tuple=True)
            # noteOutputs, _ = tf.nn.dynamic_rnn(noteStack, modelInput, dtype=tf.float32, time_major=True)
            timeFin = tf.reshape(timeOutputs, [batch_len-1,batch_width,128,300])
            timeFin = tf.transpose(timeFin, [2,1,0,3])
            timeFin = tf.reshape(timeFin, [128,batch_width*(batch_len-1),300])

            tmp = numpy.zeros(shape=(1,batch_width*(batch_len-1),2))
            tmp = tf.stack(tmp)
            tmp = tf.cast(tmp, tf.float32)

            actual_note = tf.transpose(batch[:,1:,0:-1,:], [2,0,1,3])
            actual_note = tf.reshape(actual_note, [127,batch_width*batch_width,2])
            actual_note = tf.concat([tmp,actual_note],0)

            noteLayer_input = tf.concat([timeFin,actual_note],2)

            with tf.variable_scope("note"):
                noteStack = tf.contrib.rnn.MultiRNNCell([LSTMCell(noteNeurons[i],state_is_tuple=True) for i in range(noteLayers)], state_is_tuple=True)
                noteOutputs, _ = tf.nn.dynamic_rnn(noteStack, noteLayer_input, dtype=tf.float32, time_major=True)


            # tf.fill()

            # print("-----------")
            # print(actual_note)
            # print("-----------")
            # print(d)

            # cross_entropy = tf.reduce_mean(-tf.reduce_sum(modelInput * tf.log(y), reduction_indices=[1]))
            # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=[1.0], logits=[1.0]))
            # train_step = 0

            sess.run(tf.global_variables_initializer())
            for i in range(iterations):
                inputBatch, inputModelInput = getModelInputs()


                # train_step.run([optimizer,cross_entropy],feed_dict={batch: inputBatch, modelInput:inputModelInput})
                # train_step.run(feed_dict={batch: inputBatch, modelInput:inputModelInput})
                # sess.run([timeOutputs],feed_dict={batch: inputBatch, modelInput:inputModelInput})
                print("==================================")
                print("==================================")
                print("==================================")
                print(sess.run([noteLayer_input],feed_dict={batch: inputBatch, modelInput:inputModelInput})[0].shape)
                # print(val.eval())

            sess.close()
