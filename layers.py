import tensorflow as tf
import numpy as np
import functions

def spatial_concat(inputs):
    inputs_list = tf.unstack(inputs, axis=1)
    concat_states = []
    for i in range(len(inputs_list)):
        if i == 0:
            concat_state = tf.concat([inputs_list[i], inputs_list[i], inputs_list[i+1]], axis=-1)
        elif i == len(inputs_list)-1:
            concat_state = tf.concat([inputs_list[i-1], inputs_list[i], inputs_list[i]], axis=-1)
        else:
            concat_state = tf.concat([inputs_list[i-1], inputs_list[i], inputs_list[i+1]], axis=-1)
        concat_states.append(concat_state)
    concat_states = tf.stack(concat_states, axis=1)
    return concat_states

def attention(inputs, W, v, input_lengths, hidden_size):
    inputs_W = tf.tanh(tf.reshape(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W),
                                  [-1, tf.shape(inputs)[1], hidden_size]))
    attn_weights = tf.reduce_sum(tf.multiply(v, inputs_W), 2)
    weights_mask = tf.sequence_mask(input_lengths, maxlen=tf.to_int32(tf.shape(attn_weights)[1]), dtype=tf.float32)
    attn_weights = attn_weights * weights_mask + ((1.0 - weights_mask) * tf.float32.min)
    attn_weights = tf.nn.softmax(attn_weights)
    inputs_attn = tf.multiply(tf.expand_dims(attn_weights, 2), inputs)
    attn_state = tf.reshape(tf.reduce_sum(inputs_attn, 1), [-1, hidden_size])
    return attn_state

def embeddings_layer(vocab_size, embedding_dim, trainable=True, pretrained_embeddings=None, name='embeddings'):
    if pretrained_embeddings is not None:
        embeddings = tf.get_variable(shape=pretrained_embeddings.shape,
                                     initializer=tf.constant_initializer(pretrained_embeddings),
                                     trainable=trainable,
                                     name=name)
    else:
        embeddings = tf.Variable(tf.truncated_normal([vocab_size, embedding_dim], stddev=0.1),
                                 name=name)
    return embeddings