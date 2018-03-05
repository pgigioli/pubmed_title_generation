import tensorflow as tf
import layers
import cnn

def gather_state(states, max_lens, hidden_size):
    indices = tf.range(tf.shape(states)[0])*tf.shape(states)[1] + (max_lens - 1)
    return tf.gather(tf.reshape(states, [-1, hidden_size]), indices)

def get_final_lstm_state(states, max_lens, hidden_size):
    c_states, h_states = zip(*states)
    return tf.contrib.rnn.LSTMStateTuple(gather_state(tf.stack(c_states, axis=1), max_lens, hidden_size), 
                                         gather_state(tf.stack(h_states, axis=1), max_lens, hidden_size))

def get_final_gru_state(states, max_lens, hidden_size):
    return gather_state(tf.stack(states, axis=1), max_lens, hidden_size)
    
def get_final_multi_state(states, max_lens, hidden_size):
    states = list(zip(*states))
    final_state = []
    for layer in states:
        if type(layer[0]) == tf.contrib.rnn.LSTMStateTuple:
            final_state.append(get_final_lstm_state(layer, max_lens, hidden_size))
        else:
            final_state.append(get_final_gru_state(layer, max_lens, hidden_size))
    return tuple(final_state)

def RNN_encoder(inputs, cell, hidden_size, init_state=None, input_lens=None):
    if init_state is None:
        init_state = cell.zero_state(tf.shape(inputs)[0], tf.float32)
    state = init_state
    outputs = []
    states = []
    inputs = tf.unstack(inputs, axis=1)
    for i in range(len(inputs)):
        if i > 0:
            tf.get_variable_scope().reuse_variables()
        prev_word = inputs[i]
        h, state = cell(prev_word, state)
        outputs.append(h)
        states.append(state)
    outputs = tf.stack(outputs, axis=1)
    
    if input_lens is not None:
        final_output = gather_state(outputs, input_lens, hidden_size)
        if type(states[0]) == tuple:
            final_state = get_final_multi_state(states, input_lens, hidden_size)
        elif type(states[0]) == tf.contrib.rnn.LSTMStateTuple:
            final_state = get_final_lstm_state(states, input_lens, hidden_size)
        else:
            final_state = get_final_gru_state(states, input_lens, hidden_size)
    else:
        final_output = h
        final_state = state
    return outputs, final_output, final_state