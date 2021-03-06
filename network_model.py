from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, BasicLSTMCell
import tensorflow as tf
from train_model import getModelInputs, batchToVectors, batch_len, batch_width, stateToInputVectorArray
import numpy as numpy
import tflearn as tflearn
import json


numpy.set_printoptions(threshold=numpy.nan)

def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
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
      with tf.name_scope('sigweights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('sigbias'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        tensDot = tf.tensordot(input_tensor, weights, [[2],[0]])
      return act((tensDot+biases), name='activation')

def trainingInputs(batch):
    # creates a numpy array that will match our input into the note layer
    # (1,batch*time,2)
    with tf.name_scope('casting'):
        # tmp = numpy.zeros(shape=(1,batch_width*(batch_len-1),2))
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
    return actual_note


def predInputs(batch):
    # actual_note = tf.reshape(batch, [128,1,2])
    tmp = numpy.zeros(shape=(1,1,2))
    tmp = tf.stack(tmp)
    tmp = tf.cast(tmp, tf.float32)


# next we take the actual notes that were played in the form (batch, time, notes, 2) and ignore the first time: 0
# we ignore time: 0 because we are passing in the notes that where actually played at the next time step
# we then transpose that to (notes,batch,time,2) -> (127, batch*time,2)
# then pad the first layer with 0s from tmp to get (notes,time*batch,2)
# now our actual_notes match the dimensions of our timeFin dimensions
    with tf.name_scope('reshaping'):
        actual_note = tf.transpose(batch[:,:,0:-1,:], [2,0,1,3])
        actual_note = tf.reshape(actual_note, [127,1,2])
        actual_note = tf.concat([tmp,actual_note],0)
    return actual_note


class choraleModel(object):

    def __init__(self, is_training, timeNeurons, timeLayers, noteNeurons, noteLayers, dropout):

        iterations = 5000;

        with tf.Session() as sess:

            is_training = tflearn.get_training_mode()

            with tf.name_scope('inputs'):
                batch = tf.placeholder(tf.float32, [None, None, 128, 2]) # (batch, time, notes, 2)
                modelInput = tf.placeholder(tf.float32, [None, None, 80]) # subtract the last layer we don't need its output (time-1, batch*notes, vector len)
                # Tensorflow needs 3d array so modelInput [[vector],[vector],[vector]->128*batch],[[vector],[vector],[vector]->128*batch]
                #                                          ------------------------------------time--------------------------------------

            with tf.variable_scope("time"):
                # LSTMCell makes a whole layer, in this case with 300 neurons, and timeStack is both of the layers
                timeStack = tf.contrib.rnn.MultiRNNCell([LSTMCell(timeNeurons[0],state_is_tuple=True) for _ in range(timeLayers)], state_is_tuple=True)
                #we pass our layers to dynamic_rnn and it loops through our modelInput and trains the weights
                timeOutputs, _ = tf.nn.dynamic_rnn(timeStack, modelInput, dtype=tf.float32, time_major=True)

            # timeOutputs is the outputs at the last timestep, current of shape (time-1, batch*notes, vector len)
            # In order to now loop throguh the notes in the note layer we need to get (note,time*batch,hiddens)
            # first reshape to (time,batch,notes,hiddens) -> (notes, batch, time, hiddens) -> (note, time*batch,hiddens)
            with tf.name_scope('reshaping'):
                # timeFin = tf.reshape(timeOutputs, [batch_len-1,batch_width,128,300])
                timeFin = tf.reshape(timeOutputs, [tf.shape(modelInput)[0],tf.shape(batch)[0],128,300])
                timeFin = tf.transpose(timeFin, [2,1,0,3])
                # timeFin = tf.reshape(timeFin, [128,batch_width*(batch_len-1),300])
                timeFin = tf.reshape(timeFin, [128,tf.shape(modelInput)[0]*tf.shape(batch)[0],300])

            actual_note = tf.cond(is_training, lambda: trainingInputs(batch), lambda: predInputs(batch))

            # we take the timeFin and smoosh it with the actual notes played along the 2 axis
            # this means that the actual notes played are added with the 300 hiddens we are passing in
            # (notes,batch*time,hiddens+batchs)
            noteLayer_input = tf.concat([timeFin,actual_note],2)

            # This is the start of the note layer, we are passing (note, batch*time, hiddens)
            with tf.variable_scope("note"):
                noteStack = tf.contrib.rnn.MultiRNNCell([LSTMCell(noteNeurons[i],state_is_tuple=True) for i in range(noteLayers)], state_is_tuple=True)
                noteOutputs, _ = tf.nn.dynamic_rnn(noteStack, noteLayer_input, dtype=tf.float32, time_major=False)

            # we now pass noteOutputs to a sigmoid layer
            # noteOutputs is (note, batch*time, hiddens) -> (note, batch*time, 2)
            # 2 represents [playProb,ArticProb]
            sig_layer = nn_layer(noteOutputs, 50 , 2 , layer_name="sigmoid")

            # NoteFin takes in the sigmoid layer (note, batch*time, 2) and reshapes it to (note, batch, time, 2)
            # then transpose to be (batch,time,notes,2), which matches the shape of our batch inputs
            noteFin = tf.reshape(sig_layer, [128,batch_width,(batch_len-1),2])
            noteFin= tf.transpose(noteFin, [1,2,0,3])

            # actual_note are the notes played at the next time step, we used this in addition to the time layer to pass into the note layer
            # We now need this same input to test what comes out of the note layer
            # here we are masking out the articulation prob
            actualPlayProb = actual_note[:,:,0:1]

            # we mask out the articulation prob like we did for the actual_note
            playProb = sig_layer[:,:,0:1]

            # the mask has all 1s for the play prob and 1s for the articProb if those notes where actually being played in the input batch
            # we can multiple this by what we guess to get a more acc representation of our model predictions
            actual_note_padded = tf.expand_dims(batch[:,1:,:,0],3)
            mask = tf.concat([tf.ones_like(actual_note_padded, optimize=True),actual_note_padded],axis=3)

            # sometimes the cross entropy will go to 0 if epsilon is too small or there is no epsilon
            eps = tf.constant(1.0e-7)

            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=playProb,labels=actualPlayProb))
            # this the cross entropy function
            percentages = mask * tf.log( 2 * noteFin * batch[:,1:] - noteFin - batch[:,1:] + 1 + eps )
            cost = tf.negative(tf.reduce_sum(percentages))

            # optimizer
            train_step = tf.train.RMSPropOptimizer(0.01).minimize(cost)
            # train_step = tf.train.AdadeltaOptimizer(learning_rate=0.1, epsilon=1e-6).minimize(cost)

            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter('~.', sess.graph)

            f = open('training_results.txt', 'w')

            tflearn.is_training(True)

            for i in range(iterations):
                inputBatch, inputModelInput = getModelInputs()
                # print(inputBatch)
                # print(inputModelInput)
                if i % 50 == 1:
                    train_accuracy = cost.eval(feed_dict={batch: inputBatch, modelInput:inputModelInput})
                    print("step %d, training cost %g"%(i, train_accuracy))
                    f.write("step %d, training cost %g\n"%(i, train_accuracy))
                train_step.run(feed_dict={batch: inputBatch, modelInput:inputModelInput})
                # if i > (iterations-100):
                #     an = sess.run([sig_layer],feed_dict={batch: inputBatch, modelInput:inputModelInput})
                #
                #     result = tf.round(an[0]).eval()
                #     result = tf.transpose(result, [1,0,2]).eval()
                #     for k in range(128):
                #         if result[0,k,0] == 0:
                #             result[0,k,1] = 0
                #     print("======")
                #     print(result)
                #     print("___________")
                #     song = numpy.append(song, result, axis=0)


                merged = tf.summary.merge_all()
                train_writer.add_summary(merged,i)



            tflearn.is_training(False)
            f.close()
            print("=======================================================")


            song = numpy.array(numpy.zeros([1,128,2]))

            startBatchInput = numpy.array(numpy.zeros([1,1,128,2]))
            preinput = numpy.array(numpy.zeros([1,128,2])).tolist()

            startModelInput = numpy.array(batchToVectors(preinput))
            prev = startBatchInput

            for j in range(500):

                result = sess.run([sig_layer],feed_dict={batch: startBatchInput, modelInput:startModelInput})[0]

                startBatchInput = prev
                # print(result)
                # result = tf.nn.softmax(result)
                result = tf.round(result)

                result = tf.transpose(result, [1,0,2]).eval()
                # print(result.shape)
                for k in range(128):
                    if result[0,k,0] == 0:
                        result[0,k,1] = 0
                startModelInput = numpy.array([stateToInputVectorArray(j,state) for _,state in enumerate(result.tolist())])

                song = numpy.append(song, result, axis=0)
                prev = numpy.reshape(result, [1,1,128,2])
                # startModelInput = numpy.array([stateToInputVectorArray(j,state) for _,state in enumerate(result.tolist())])
                # startModelInput =  numpy.array(batchToVectors(result))
            # print(batch)

            matDict = dict()
            matDict["test"] = song.tolist()

            dataJSON = open('dataJSON.json', 'w')
            json.dump(matDict, dataJSON)
            dataJSON.close()

            saver.save(sess,'./train_model_checkpoint/export-model')
            sess.close()
