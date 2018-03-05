import tensorflow as tf
import numpy as np

def masked_log_loss(softmax, targets, target_lens, batch_size):
    seq_len = tf.shape(targets)[1]
    softmax_list = tf.unstack(softmax, axis=1)
    targets_list = tf.unstack(targets, axis=1)
    batch_nums = tf.range(0, batch_size)
    losses = []
    for i, softmax in enumerate(softmax_list):
        negative_log = -tf.log(softmax+1e-12)
        indices = tf.stack((batch_nums, targets_list[i]), axis=1)
        step_loss = tf.gather_nd(negative_log, indices)
        losses.append(step_loss)
    losses = tf.stack(losses, axis=1)
    mask = tf.sequence_mask(target_lens, maxlen=seq_len)
    masked_loss = tf.to_float(mask)*losses
    total_loss = tf.reduce_mean(tf.reduce_sum(masked_loss, axis=1) / tf.to_float(target_lens)) 
    return total_loss

def compute_masked_accuracy(predictions, targets, target_lens):
    mask = tf.sequence_mask(target_lens, maxlen=tf.shape(targets)[1])
    equal = tf.equal(targets, predictions)
    masked_equal = tf.multiply(tf.to_float(equal), tf.to_float(mask))
    masked_equal_sum = tf.reduce_sum(tf.to_float(masked_equal), axis=1)
    batch_acc = tf.divide(masked_equal_sum, tf.to_float(target_lens))
    total_acc = tf.reduce_mean(batch_acc)
    return total_acc

def count_params(variables):
    total_parameters = 0
    for variable in variables:
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('# Trainable Parameters: {}'.format(total_parameters))
    
def write_to_log(string, filename):
    with open(filename, 'a') as write_file:
        write_file.write(string + '\n')

def create_embeddings(vocab, w2v):
    embeddings = np.zeros((vocab.num_words, w2v.vector_size))
    w2v_ct = 0
    zero_ct = 0
    for key, value in vocab.get_word2id_items():
        try:
            embeddings[value] = w2v[key]
            w2v_ct += 1
        except:
            embeddings[value] = np.random.uniform(-0.1, 0.1, w2v.vector_size)
            zero_ct += 1
    print("{} words initialized from word2vec, {} words randomly initialized".format(w2v_ct, zero_ct))
    return embeddings