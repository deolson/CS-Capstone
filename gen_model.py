from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, BasicLSTMCell
import tensorflow as tf
from train_model import getModelInputs, batchToVectors, batch_len, batch_width, stateToInputVectorArray
import numpy as numpy
import tflearn as tflearn
import json
import numpy.ma as ma

numpy.set_printoptions(threshold=numpy.nan)


#defines weights given a matrix dimension
def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)

#defines bias given a matrix dimension
def bias_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.01)
    return tf.Variable(initial)

#used for tensor board visualization
def variable_summaries(var):
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

#nn_layer lets you define an activation layer
#we use this for a custom sigmoid layer, but this can be used for any activation function
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.sigmoid):
    with tf.name_scope(layer_name):
      with tf.name_scope('sigweights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('sigbias'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        # this applies the activation to the created weights and biases
        tensDot = tf.tensordot(input_tensor, weights, [[2],[0]])
      return act((tensDot+biases), name='activation')

def trainingInputs(batch):

    # creates a numpy array of 0s that will match our input into the note layer
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
    return actual_note

# when we are gen the song we need to manually adjust the batch vector because
# its dimensions no longer fit to our training dimensions
def predInputs(batch):
    actual_note = tf.reshape(batch, [128,1,2])
    return actual_note


class choraleModel(object):

    def __init__(self, is_training, timeNeurons, timeLayers, noteNeurons, noteLayers, dropout):
        #this iterations defines how long the network will train for
        iterations = 2000;

        with tf.Session() as sess:
            # you can't reg variables to define network behavior, tf provides options for boolean logic though
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

            # optimizer - goes and adjusts our defined weights in backprop - trying to reduce cost
            train_step = tf.train.RMSPropOptimizer(0.01).minimize(cost)
            # train_step = tf.train.AdadeltaOptimizer(learning_rate=0.1, epsilon=1e-6).minimize(cost)


            #saver lets us take snapshots of the network, sess run is a tf struct that needs to be called
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter('~.', sess.graph)

            # training results is used for debugging/ checking our cost function
            # f = open('training_results.txt', 'w')


            # now we actually start training
            tflearn.is_training(True)
            song = numpy.array(numpy.zeros([1,128,2]))
            for i in range(iterations):

                # this grabs random sections in our song dictionary that we then train the network on
                inputBatch, inputModelInput = getModelInputs()
                # print(inputBatch)
                # print(inputModelInput)

                # looks at the cost function as the networks training
                if i % 20 == 1:
                    train_accuracy = cost.eval(feed_dict={batch: inputBatch, modelInput:inputModelInput})
                    print("step %d, training cost %g"%(i, train_accuracy))

                    # writing to our debug file
                    # f.write("step %d, training cost %g\n"%(i, train_accuracy))


                # this is were the magic happens, this takes our song section and feeds it into the network
                train_step.run(feed_dict={batch: inputBatch, modelInput:inputModelInput})

                # DEBUGGING PURPOSES DO NOT REMOVE
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


            # At this point the network has been trained and all the weights are set
            # We now need to actually start generation
            tflearn.is_training(False)
            print("=======================================================")


            # we need to input both the batch adn the model input, however because we are now generating
            # we no longer use the batch inside the model -- hence the need for the tf.learn.is_training
            startBatchInput = numpy.array(numpy.zeros([1,1,128,2]))
            preinput = numpy.array(numpy.zeros([1,128,2])).tolist()

            startModelInput = numpy.array(batchToVectors(preinput))
            prev = startBatchInput

            # range will define how long the song will be
            for j in range(500):

                # we feed the input into the network, but instead of going all the way to the end with the cost function
                # we only go until the sig_layer and take that output to be our notes that get played
                result = sess.run([sig_layer],feed_dict={batch: startBatchInput, modelInput:startModelInput})

                startBatchInput = prev
                # print(result)

                # we round the % values to being played or not played / 0 or 1
                # note under 50% are not played, notes above 50% do get played
                result = tf.round(result[0]).eval()

                result = tf.transpose(result, [1,0,2]).eval()

                # this is taking care of the articProb
                for k in range(128):
                    if result[0,k,0] == 0:
                        result[0,k,1] = 0

                # at this point our result no fits the structure of our statematrices and we can append and generate the next timestep
                song = numpy.append(song, result, axis=0)
                prev = numpy.reshape(result, [1,1,128,2])
                startModelInput = numpy.array([stateToInputVectorArray(j,state) for _,state in enumerate(result.tolist())])

            # print(batch)

            # we now start the process of converting back to python 2.7 to construct our midi
            matDict = dict()
            matDict["test"] = song.tolist()

            dataJSON = open('dataJSON.json', 'w')
            json.dump(matDict, dataJSON)
            dataJSON.close()

            saver.save(sess,'./train_model_checkpoint/export-model')
            sess.close()
