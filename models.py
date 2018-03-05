import tensorflow as tf
import layers
import functions
import decoder
import encoder

class RNN_Classifier:
    def __init__(self, n_classes, vocab_size, max_len, embedding_dim=100, hidden_size=128, n_layers=1, 
                 bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True):
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.trainable_embeddings = trainable_embeddings
        self._pretrained_embeddings = pretrained_embeddings
        
        self.inputs = tf.placeholder(tf.int32, [None, self.max_len], name='inputs')
        self.input_lens = tf.placeholder(tf.int32, [None], name='input_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        
        with tf.variable_scope('embeddings_layer'):
            self.embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                      trainable=self.trainable_embeddings,
                                                      pretrained_embeddings=self._pretrained_embeddings,
                                                      name='embeddings')
            inputs_embd = tf.nn.embedding_lookup(self.embeddings, self.inputs)

        def gru_cell():
            return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(num_units=self.hidden_size), 
                                                 output_keep_prob=self.dropout_keep_prob)
            
        with tf.variable_scope('encoder'):
            if self.n_layers > 1:
                self.cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(self.n_layers)])
            else:
                self.cell = gru_cell()
            outputs, final_output, final_state = encoder.RNN_encoder(inputs_embd, self.cell, self.hidden_size, 
                                                                     input_lens=self.input_lens)
            
        with tf.variable_scope('dense_output'):
            self.output_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.n_classes]), name='output_W')
            self.output_b = tf.Variable(tf.constant(0.0, shape=[self.n_classes]), name='output_b')
            
            self.logits = tf.matmul(final_output, self.output_W) + self.output_b
            
class Seq2Seq_Basic_Attn:
    def __init__(self, vocab_size, max_enc_len, max_dec_len, embedding_dim=100, hidden_size=128, 
                 n_layers=1, bidirectional=False, pretrained_embeddings=None, trainable_embeddings=True, 
                 shared_embeddings=True, weight_tying=False, rnn_cell=tf.contrib.rnn.GRUCell):
        self.vocab_size = vocab_size
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.trainable_embeddings = trainable_embeddings
        self.shared_embeddings = shared_embeddings
        self.weight_tying = weight_tying
        self.rnn_cell = rnn_cell
        self._pretrained_embeddings = pretrained_embeddings
        
        self.enc_inputs = tf.placeholder(tf.int32, [None, self.max_enc_len], name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, self.max_dec_len], name='dec_inputs')
        self.enc_lens = tf.placeholder(tf.int32, [None], name='enc_lens')
        self.dec_lens = tf.placeholder(tf.int32, [None], name='dec_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.teacher_forcing = tf.placeholder(tf.bool, name='teacher_forcing')
        self.teacher_forcing_mask = tf.placeholder(tf.int32, [self.max_dec_len], name='teacher_forcing_mask')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.sample_decoding = tf.placeholder(tf.bool, name='sample_decoding')
                
        with tf.variable_scope('embeddings_layer'):
            if self.shared_embeddings == True:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='embeddings')
                self.dec_embeddings = self.enc_embeddings
            else:
                self.enc_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='enc_embeddings')
                self.dec_embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                              trainable=self.trainable_embeddings,
                                                              pretrained_embeddings=self._pretrained_embeddings,
                                                              name='dec_embeddings')
            enc_inputs_embd = tf.nn.embedding_lookup(self.enc_embeddings, self.enc_inputs)
            dec_inputs_embd = tf.nn.embedding_lookup(self.dec_embeddings, self.dec_inputs)
        
        def cell():
            return tf.contrib.rnn.DropoutWrapper(self.rnn_cell(num_units=self.hidden_size), 
                                                 output_keep_prob=self.dropout_keep_prob)
        
        with tf.variable_scope('encoder'):
            if self.n_layers > 1:
                self.enc_cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(self.n_layers)])
            else:
                self.enc_cell = cell()
            enc_outputs, final_output, final_state = encoder.RNN_encoder(enc_inputs_embd, self.enc_cell, 
                                                                         self.hidden_size, input_lens=self.enc_lens)
            
        with tf.variable_scope('attention'):
            self.attn_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.hidden_size]), name='attn_W')
            self.attn_v = tf.Variable(tf.truncated_normal([self.hidden_size]), name='attn_v')
            attn_state = layers.attention(enc_outputs, self.attn_W, self.attn_v, self.enc_lens, self.hidden_size)
            
        with tf.variable_scope('decoder'):
            if self.n_layers > 1:
                self.dec_cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(self.n_layers)])
            else:
                self.dec_cell = cell()
            if self.weight_tying == True:
                proj_W = tf.Variable(tf.truncated_normal([2*self.hidden_size, self.embedding_dim]), name='proj_W')
                self.dec_W = tf.tanh(tf.matmul(proj_W, tf.transpose(self.dec_embeddings, [1, 0])))
            else:
                self.dec_W = tf.Variable(tf.truncated_normal([2*self.hidden_size, self.vocab_size]), name='dec_W')
            self.dec_b = tf.Variable(tf.constant(0.0, shape=[self.vocab_size]), name='dec_b')
            self.go_var = tf.Variable(tf.truncated_normal([self.embedding_dim]), name='go_var')

            (self.logits, self.dec_states, 
             self.generated_words) = decoder.RNN_basic_attn_decoder(dec_inputs_embd, self.dec_cell, self.go_var, self.dec_W, 
                                                                    self.dec_b, self.dec_embeddings, final_state, attn_state, 
                                                                    self.hidden_size, teacher_forcing=self.teacher_forcing, 
                                                                    teacher_forcing_mask=self.teacher_forcing_mask, 
                                                                    sample_decoding=self.sample_decoding)