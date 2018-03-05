import tensorflow as tf
import random 
import layers

def RNN_decoder(dec_inputs, cell, go_var, output_W, output_b, embeddings, init_state, hidden_size, teacher_forcing=False,
                teacher_forcing_mask=None, sample_decoding=False):
    if teacher_forcing_mask is None:
        teacher_forcing_mask = tf.ones([tf.shape(dec_inputs)[1]])
    batch_size = tf.shape(dec_inputs)[0]
        
    dec_inputs = tf.unstack(dec_inputs, axis=1)
    h_states = []
    output_logits = []
    generated_words = []
    for i in range(len(dec_inputs)): 
        if i == 0:
            prev_word = tf.tile(tf.expand_dims(go_var, 0), [batch_size, 1])
            h, state = cell(prev_word, init_state)
        else:
            tf.get_variable_scope().reuse_variables()
            one_or_zero = tf.equal(teacher_forcing_mask[i], 1)
            teacher_forcing_step = tf.cond(one_or_zero, lambda: teacher_forcing, lambda: False)
            generated_word = tf.cond(sample_decoding, lambda: tf.to_int32(tf.squeeze(tf.multinomial(logits, 1), axis=1)), 
                                     lambda: tf.to_int32(tf.argmax(logits, axis=1)))
            prev_word = tf.cond(teacher_forcing_step, lambda: dec_inputs[i-1], 
                                lambda: tf.nn.embedding_lookup(embeddings, generated_word))

            h, state = cell(prev_word, state)
            generated_words.append(generated_word)
        h_states.append(h)  
        logits = tf.matmul(h, output_W) + output_b
        output_logits.append(logits)
    output_logits = tf.stack(output_logits, axis=1)
    h_states = tf.stack(h_states, axis=1)
    
    generated_word = tf.cond(sample_decoding, lambda: tf.to_int32(tf.squeeze(tf.multinomial(logits, 1), axis=1)), 
                             lambda: tf.to_int32(tf.argmax(logits, axis=1)))
    generated_words.append(generated_word)
    generated_words = tf.stack(generated_words, axis=1)
    return output_logits, h_states, generated_words 

def RNN_basic_attn_decoder(dec_inputs, cell, go_var, output_W, output_b, embeddings, init_state, attn_state, hidden_size, 
                           teacher_forcing=False, teacher_forcing_mask=None, sample_decoding=False):
    if teacher_forcing_mask is None:
        teacher_forcing_mask = tf.ones([tf.shape(dec_inputs)[1]])
    batch_size = tf.shape(dec_inputs)[0]

    dec_inputs = tf.unstack(dec_inputs, axis=1)
    h_states = []
    output_logits = []
    generated_words = []
    for i in range(len(dec_inputs)): 
        if i > 0:
            tf.get_variable_scope().reuse_variables()
            
        if i == 0:
            prev_word = tf.tile(tf.expand_dims(go_var, 0), [batch_size, 1])
            h, state = cell(prev_word, init_state)
        else:
            one_or_zero = tf.equal(teacher_forcing_mask[i], 1)
            teacher_forcing_step = tf.cond(one_or_zero, lambda: teacher_forcing, lambda: False)
            generated_word = tf.cond(sample_decoding, lambda: tf.to_int32(tf.squeeze(tf.multinomial(logits, 1), axis=1)), 
                                     lambda: tf.to_int32(tf.argmax(logits, axis=1)))
            prev_word = tf.cond(teacher_forcing_step, lambda: dec_inputs[i-1], 
                                lambda: tf.nn.embedding_lookup(embeddings, generated_word))
            
            h, state = cell(prev_word, state)
            generated_words.append(generated_word)
        h_states.append(h)  
        logits = tf.matmul(tf.concat([h, attn_state], axis=1), output_W) + output_b
        output_logits.append(logits)
    output_logits = tf.stack(output_logits, axis=1)
    h_states = tf.stack(h_states, axis=1)
    
    generated_word = tf.cond(sample_decoding, lambda: tf.to_int32(tf.squeeze(tf.multinomial(logits, 1), axis=1)), 
                             lambda: tf.to_int32(tf.argmax(logits, axis=1)))
    generated_words.append(generated_word)
    generated_words = tf.stack(generated_words, axis=1)
    return output_logits, h_states, generated_words   