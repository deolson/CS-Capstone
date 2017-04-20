from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, BasicLSTMCell
import tensorflow as tf
from train_model import getModelInputs, batch_len, division_len, binary_len, batch_width
import numpy as numpy
numpy.set_printoptions(threshold=numpy.nan)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.01)
    return tf.Variable(initial)

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

class choraleModel(object):

    def __init__(self, is_training, timeNeurons, timeLayers, noteNeurons, noteLayers, dropout):

        iterations = 101;

        with tf.Session() as sess:
            # inputs to the model, batch is a submatrix of the statematrix that is fitted to our batchs and modelInput is the note vectors
            with tf.name_scope('inputs'):
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
            with tf.name_scope('reshaping'):
                timeFin = tf.reshape(timeOutputs, [batch_len-1,batch_width,128,300])
                timeFin = tf.transpose(timeFin, [2,1,0,3])
                timeFin = tf.reshape(timeFin, [128,batch_width*(batch_len-1),300])

            # creates a numpy array that will match our input into the note layer
            # (1,batch*time,2)
            with tf.name_scope('casting'):
                tmp = numpy.zeros(shape=(1,batch_width*(batch_len-1),2))
                tmp = tf.stack(tmp)
                tmp = tf.cast(tmp, tf.float32)

            # next we take the actual notes that were played in the form (batch, time, notes, 2) and ignore the first time: 0
            # we ignore time: 0 because we are passing in the notes that where actually played at the next time step
            # we then transpose that to (notes,batch,time,2) -> (127, batch*time,2)
            # then pad the first layer with 0s from tmp to get (notes,time*batch,2)
            # now our actual_notes match the dimensions of our timeFin dimensions
            with tf.name_scope('reshaping'):
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


            sig_layer = nn_layer(noteOutputs, 50 , 2 , layer_name="sigmoid")

            noteFin = tf.reshape(sig_layer, [128,batch_width,(batch_len-1),2])
            noteFin= tf.transpose(noteFin, [1,2,0,3])

            actualPlayProb = actual_note[:,:,0:1]
            # actualPlayProb = tf.reshape(actualPlayProb, [128*(batch_len-1)*batch_width,1])
            # actualArticProb = sig_layer[:,:,1:]

            playProb = sig_layer[:,:,0:1]
            # playProb = tf.reshape(playProb, [128*(batch_len-1)*batch_width,1])

            articProb = sig_layer[:,:,1:]

            actual_note_padded = tf.expand_dims(batch[:,1:,:,0],3)
            mask = tf.concat([tf.ones_like(actual_note_padded, optimize=True),actual_note_padded],axis=3)
            eps = tf.constant(1.19209e-07)
            percentages = mask * tf.log( 2 * noteFin * batch[:,1:] - noteFin - batch[:,1:] + 1 + eps )
            cost = tf.negative(tf.reduce_sum(percentages))

            # softmax = tf.nn.softmax_cross_entropy_with_logits(logits=playProb,labels=actualPlayProb)
            # softmax = tf.nn.softmax(sig_layer)


            # cost = tf.reduce_mean(cost)
            train_step = tf.train.RMSPropOptimizer(0.001).minimize(cost)
            # train_step = tf.train.AdadeltaOptimizer(learning_rate=0.1, epsilon=1e-6).minimize(cost)

            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter('~.', sess.graph)

            for i in range(iterations):
                inputBatch, inputModelInput = getModelInputs()
                if i % 10 == 1:
                    train_accuracy = cost.eval(feed_dict={batch: inputBatch, modelInput:inputModelInput})
                    print("step %d, training accuracy %g"%(i, train_accuracy))
                train_step.run(feed_dict={batch: inputBatch, modelInput:inputModelInput})
                if i == (iterations-1):
                    an = sess.run([actualPlayProb],feed_dict={batch: inputBatch, modelInput:inputModelInput})
                    ad = sess.run([playProb],feed_dict={batch: inputBatch, modelInput:inputModelInput})
                    print(an[0])
                    print(ad[0])

                merged = tf.summary.merge_all()
                train_writer.add_summary(merged,i)

            saver = tf.train.Saver()

            saver.save(sess,'~./model.ckpt', i)
            sess.close()
