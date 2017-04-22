from train_model import getModelInputs, getFirstRowPlayed, batch_len, batch_width
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, BasicLSTMCell
import tensorflow as tf
import numpy as numpy
numpy.set_printoptions(threshold=numpy.nan)

timeNeurons = [300,300]
timeLayers = 2
noteNeurons = [100,50]
noteLayers = 2


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name= 'sigweights')

def bias_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.01)
    return tf.Variable(initial, name='sigbias')

def variable_summaries(var):
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.sigmoid):
    with tf.name_scope(layer_name):
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        tensDot = tf.tensordot(input_tensor, weights, [[2],[0]])
      return act((tensDot+biases), name='activation')



with tf.Session() as sess:

    with tf.name_scope('inputs'):
        batch = tf.placeholder(tf.float32, [batch_width, batch_len, 128, 2])
        modelInput = tf.placeholder(tf.float32, [batch_len-1, batch_width*128, 80])

    with tf.variable_scope("time"):
        timeStack = tf.contrib.rnn.MultiRNNCell([LSTMCell(timeNeurons[0],state_is_tuple=True) for _ in range(timeLayers)], state_is_tuple=True)
        timeOutputs, _ = tf.nn.dynamic_rnn(timeStack, modelInput, dtype=tf.float32, time_major=True)

    with tf.name_scope('reshaping'):
        timeFin = tf.reshape(timeOutputs, [batch_len-1,batch_width,128,300])
        timeFin = tf.transpose(timeFin, [2,1,0,3])
        timeFin = tf.reshape(timeFin, [128,batch_width*(batch_len-1),300])

    with tf.name_scope('casting'):
        tmp = numpy.zeros(shape=(1,batch_width*(batch_len-1),2))
        tmp = tf.stack(tmp)
        tmp = tf.cast(tmp, tf.float32)

    with tf.name_scope('reshaping'):
        actual_note = tf.transpose(batch[:,1:,0:-1,:], [2,0,1,3])
        actual_note = tf.reshape(actual_note, [127,batch_width*(batch_len-1),2])
        actual_note = tf.concat([tmp,actual_note],0)

        noteLayer_input = tf.concat([timeFin,actual_note],2)

    with tf.variable_scope("note"):
        noteStack = tf.contrib.rnn.MultiRNNCell([LSTMCell(noteNeurons[i],state_is_tuple=True) for i in range(noteLayers)], state_is_tuple=True)
        noteOutputs, _ = tf.nn.dynamic_rnn(noteStack, noteLayer_input, dtype=tf.float32, time_major=False)

    sig_layer = nn_layer(noteOutputs, 50 , 2, layer_name="sigmoid")

    # sess.run(tf.global_variables_initializer())
    sess.run(tf.initialize_all_variables())
    saver = tf.train.import_meta_graph('./train_model_checkpoint/export-model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./train_model_checkpoint/'))
    # print(tf.trainable_variables())
    inputBatch, inputModelInput = getModelInputs()
    an = sess.run([sig_layer],feed_dict={batch: inputBatch, modelInput:inputModelInput})
    print(an[0])
