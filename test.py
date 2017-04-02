import theano, theano.tensor as T
import numpy as np
import theano_lstm

from theano_lstm import Embedding, LSTM, RNN, StackedCells, Layer, create_optimization_updates, masked_loss, MultiDropout

def has_hidden(layer):
    """
    Whether a layer has a trainable
    initial hidden state.
    """
    return hasattr(layer, 'initial_hidden_state')

def matrixify(vector, n):
    # Cast n to int32 if necessary to prevent error on 32 bit systems
    return T.repeat(T.shape_padleft(vector),
                    n if (theano.configdefaults.local_bitwidth() == 64) else T.cast(n,'int32'),
                    axis=0)

def initial_state(layer, dimensions = None):
    """
    Initalizes the recurrence relation with an initial hidden state
    if needed, else replaces with a "None" to tell Theano that
    the network **will** return something, but it does not need
    to send it to the next step of the recurrence
    """
    if dimensions is None:
        return layer.initial_hidden_state if has_hidden(layer) else None
    else:
        return matrixify(layer.initial_hidden_state, dimensions) if has_hidden(layer) else None

def initial_state_with_taps(layer, dimensions = None):
    """Optionally wrap tensor variable into a dict with taps=[-1]"""
    state = initial_state(layer, dimensions)
    if state is not None:
        return dict(initial=state, taps=[-1])
    else:
        return None

class PassthroughLayer(Layer):
    """
    Empty "layer" used to get the final output of the LSTM
    """

    def __init__(self):
        self.is_recursive = False

    def create_variables(self):
        pass

    def activate(self, x):
        return x

    @property
    def params(self):
        return []

    @params.setter
    def params(self, param_list):
        pass


def get_last_layer(result):
    if isinstance(result, list):
        return result[-1]
    else:
        return result

def ensure_list(result):
    if isinstance(result, list):
        return result
    else:
        return [result]


class Model(object):

    def __init__(self, t_layer_sizes, p_layer_sizes, dropout=0):

        self.t_layer_sizes = t_layer_sizes
        self.p_layer_sizes = p_layer_sizes

        # From our architecture definition, size of the notewise input
        self.t_input_size = 80

        # time network maps from notewise input size to various hidden sizes
        self.time_model = StackedCells( self.t_input_size, celltype=LSTM, layers = t_layer_sizes)
        self.time_model.layers.append(PassthroughLayer())

        # pitch network takes last layer of time model and state of last note, moving upward
        # and eventually ends with a two-element sigmoid layer
        p_input_size = t_layer_sizes[-1] + 2
        self.pitch_model = StackedCells( p_input_size, celltype=LSTM, layers = p_layer_sizes)
        self.pitch_model.layers.append(Layer(p_layer_sizes[-1], 2, activation = T.nnet.sigmoid))

        self.dropout = dropout

        self.conservativity = T.fscalar()
        self.srng = T.shared_randomstreams.RandomStreams(np.random.randint(0, 1024))

        self.setup_train()

    @property
    def params(self):
        return self.time_model.params + self.pitch_model.params

    @params.setter
    def params(self, param_list):
        ntimeparams = len(self.time_model.params)
        self.time_model.params = param_list[:ntimeparams]
        self.pitch_model.params = param_list[ntimeparams:]

    @property
    def learned_config(self):
        return [self.time_model.params, self.pitch_model.params, [l.initial_hidden_state for mod in (self.time_model, self.pitch_model) for l in mod.layers if has_hidden(l)]]

    @learned_config.setter
    def learned_config(self, learned_list):
        self.time_model.params = learned_list[0]
        self.pitch_model.params = learned_list[1]
        for l, val in zip((l for mod in (self.time_model, self.pitch_model) for l in mod.layers if has_hidden(l)), learned_list[2]):
            l.initial_hidden_state.set_value(val.get_value())

    def setup_train(self):

        # dimensions: (batch, time, notes, input_data) with input_data as in architecture
        self.input_mat = T.btensor4()
        # dimensions: (batch, time, notes, onOrArtic) with 0:on, 1:artic
        self.output_mat = T.btensor4()

        self.epsilon = np.spacing(np.float32(1.0))

        def step_time(in_data, *other):
            other = list(other)
            split = -len(self.t_layer_sizes) if self.dropout else len(other)
            hiddens = other[:split]
            masks = [None] + other[split:] if self.dropout else []
            new_states = self.time_model.forward(in_data, prev_hiddens=hiddens, dropout=masks)
            return new_states

        def step_note(in_data, *other):
            other = list(other)
            split = -len(self.p_layer_sizes) if self.dropout else len(other)
            hiddens = other[:split]
            masks = [None] + other[split:] if self.dropout else []
            new_states = self.pitch_model.forward(in_data, prev_hiddens=hiddens, dropout=masks)
            return new_states

        # We generate an output for each input, so it doesn't make sense to use the last output as an input.
        # Note that we assume the sentinel start value is already present
        # TEMP CHANGE: NO SENTINEL
        input_slice = self.input_mat[:,0:-1]
        n_batch, n_time, n_note, n_ipn = input_slice.shape

        # time_inputs is a matrix (time, batch/note, input_per_note)
        time_inputs = input_slice.transpose((1,0,2,3)).reshape((n_time,n_batch*n_note,n_ipn))
        num_time_parallel = time_inputs.shape[1]

        # apply dropout
        if self.dropout > 0:
            time_masks = theano_lstm.MultiDropout( [(num_time_parallel, shape) for shape in self.t_layer_sizes], self.dropout)
        else:
            time_masks = []

        time_outputs_info = [initial_state_with_taps(layer, num_time_parallel) for layer in self.time_model.layers]
        time_result, _ = theano.scan(fn=step_time, sequences=[time_inputs], non_sequences=time_masks, outputs_info=time_outputs_info)

        self.time_thoughts = time_result
        # Now time_result is a list of matrix [layer](time, batch/note, hidden_states) for each layer but we only care about
        # the hidden state of the last layer.
        # Transpose to be (note, batch/time, hidden_states)
        last_layer = get_last_layer(time_result)
        n_hidden = last_layer.shape[2]
        time_final = get_last_layer(time_result).reshape((n_time,n_batch,n_note,n_hidden)).transpose((2,1,0,3)).reshape((n_note,n_batch*n_time,n_hidden))

        # note_choices_inputs represents the last chosen note. Starts with [0,0], doesn't include last note.
        # In (note, batch/time, 2) format
        # Shape of start is thus (1, N, 2), concatenated with all but last element of output_mat transformed to (x, N, 2)
        start_note_values = T.alloc(np.array(0,dtype=np.int8), 1, time_final.shape[1], 2 )
        correct_choices = self.output_mat[:,1:,0:-1,:].transpose((2,0,1,3)).reshape((n_note-1,n_batch*n_time,2))
        note_choices_inputs = T.concatenate([start_note_values, correct_choices], axis=0)

        # Together, this and the output from the last LSTM goes to the new LSTM, but rotated, so that the batches in
        # one direction are the steps in the other, and vice versa.
        note_inputs = T.concatenate( [time_final, note_choices_inputs], axis=2 )
        num_timebatch = note_inputs.shape[1]

        # apply dropout
        if self.dropout > 0:
            pitch_masks = theano_lstm.MultiDropout( [(num_timebatch, shape) for shape in self.p_layer_sizes], self.dropout)
        else:
            pitch_masks = []

        note_outputs_info = [initial_state_with_taps(layer, num_timebatch) for layer in self.pitch_model.layers]
        note_result, _ = theano.scan(fn=step_note, sequences=[note_inputs], non_sequences=pitch_masks, outputs_info=note_outputs_info)

        self.note_thoughts = note_result

        # Now note_result is a list of matrix [layer](note, batch/time, onOrArticProb) for each layer but we only care about
        # the hidden state of the last layer.
        # Transpose to be (batch, time, note, onOrArticProb)
        note_final = get_last_layer(note_result).reshape((n_note,n_batch,n_time,2)).transpose(1,2,0,3)

        # The cost of the entire procedure is the negative log likelihood of the events all happening.
        # For the purposes of training, if the ouputted probability is P, then the likelihood of seeing a 1 is P, and
        # the likelihood of seeing 0 is (1-P). So the likelihood is (1-P)(1-x) + Px = 2Px - P - x + 1
        # Since they are all binary decisions, and are all probabilities given all previous decisions, we can just
        # multiply the likelihoods, or, since we are logging them, add the logs.

        # Note that we mask out the articulations for those notes that aren't played, because it doesn't matter
        # whether or not those are articulated.
        # The padright is there because self.output_mat[:,:,:,0] -> 3D matrix with (b,x,y), but we need 3d tensor with
        # (b,x,y,1) instead
        active_notes = T.shape_padright(self.output_mat[:,1:,:,0])
        mask = T.concatenate([T.ones_like(active_notes),active_notes], axis=3)

        loglikelihoods = mask * T.log( 2*note_final*self.output_mat[:,1:] - note_final - self.output_mat[:,1:] + 1 + self.epsilon )
        self.cost = T.neg(T.sum(loglikelihoods))

        updates, _, _, _, _ = create_optimization_updates(self.cost, self.params, method="adadelta")
        self.update_fun = theano.function(
            inputs=[self.input_mat, self.output_mat],
            outputs=self.cost,
            updates=updates,
            allow_input_downcast=True)

        self.update_thought_fun = theano.function(
            inputs=[self.input_mat, self.output_mat],
            outputs= ensure_list(self.time_thoughts) + ensure_list(self.note_thoughts) + [self.cost],
            allow_input_downcast=True)

if __name__ == '__main__':
    m = Model([300,300], [100,50], dropout=0.5)
    batch4 = np.array(([[[1,1],[1,2],[1,3],[1,4]],[[2,2],[2,2],[2,2],[2,2]],[[5,5],[5,5],[5,5],[5,5]]],[[[3,1],[3,2] ,[3,3],[3,4]], [[4,4],[4,4],[4,4],[4,4]],[[6,6],[6,6],[6,6],[6,6]]]))
    batch5 = np.array(([[[[1,1],[1,2],[1,3],[1,4]],[[2,2],[2,2],[2,2],[2,2]],[[5,5],[5,5],[5,5],[5,5]]],[[[3,1],[3,2] ,[3,3],[3,4]], [[4,4],[4,4],[4,4],[4,4]],[[6,6],[6,6],[6,6],[6,6]]]]))

    error = m.update_fun(batch5)
    print("error",error)
