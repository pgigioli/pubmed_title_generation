import tensorflow as tf
import functions

class Text_Summarization:
    def __init__(self, nn, lr=0.001, mode='train'):
        self.lr = lr
        
        self._enc_inputs = nn.enc_inputs
        self._dec_inputs = nn.dec_inputs
        self._enc_lens = nn.enc_lens
        self._dec_lens = nn.dec_lens
        self._dropout_keep_prob = nn.dropout_keep_prob
        self._teacher_forcing = nn.teacher_forcing
        self._teacher_forcing_mask = nn.teacher_forcing_mask
        self._logits = nn.logits
        self._is_training = nn.is_training
        self._sample_decoding = nn.sample_decoding
        
        self._targets = tf.placeholder(tf.int32, [None, nn.max_dec_len], name='targets')
        self._target_lens = tf.placeholder(tf.int32, [None], name='target_lens')
        self._batch_size = tf.placeholder(tf.int32, name='batch_size')
        self._loss_mask_len = tf.placeholder(tf.int32, shape=[1], name='loss_mask_len')
        self._lr = tf.placeholder(tf.float32, name='lr')
    
        self._softmax = tf.nn.softmax(self._logits)
        self._predictions = tf.to_int32(tf.argmax(self._softmax, axis=2))
        
        loss_mask = tf.minimum(self._loss_mask_len, self._target_lens)
        self._loss = functions.masked_log_loss(self._softmax, self._targets, loss_mask, self._batch_size)
                
        self._accuracy = functions.compute_masked_accuracy(self._predictions, self._targets, self._target_lens)
                
        params = tf.trainable_variables()
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self._loss, params), 1) 
        self._gradient_norm = tf.global_norm(gradients)
        
        opt_func = tf.train.AdamOptimizer(learning_rate=self._lr)
        self._optimizer = opt_func.apply_gradients(zip(gradients, params))
                
    def train_step(self, sess, enc_inputs, dec_inputs, enc_lens, dec_lens, targets, target_lens, dropout_keep_prob=1.0,
                   teacher_forcing=True, teacher_forcing_mask=None, loss_mask_len=None, sample_decoding=False, lr=None):
        if teacher_forcing_mask is None:
            teacher_forcing_mask = [1 for _ in range(len(targets[0]))]
        if lr is None:
            lr = self.lr
        if loss_mask_len is None:
            loss_mask_len = [len(targets[0])]
        else:
            loss_mask_len = [loss_mask_len]
        batch_size = len(enc_inputs)
        
        run_vars = [self._loss, self._accuracy, self._gradient_norm, self._optimizer]
        loss, accuracy, grad_norm, _ = sess.run(run_vars, feed_dict={self._enc_inputs : enc_inputs,
                                                                     self._dec_inputs : dec_inputs,
                                                                     self._enc_lens : enc_lens,
                                                                     self._dec_lens : dec_lens,
                                                                     self._targets : targets,
                                                                     self._target_lens : target_lens,
                                                                     self._dropout_keep_prob : dropout_keep_prob,
                                                                     self._teacher_forcing : teacher_forcing,
                                                                     self._teacher_forcing_mask : teacher_forcing_mask,
                                                                     self._batch_size : batch_size,
                                                                     self._loss_mask_len : loss_mask_len, 
                                                                     self._is_training : True,
                                                                     self._sample_decoding : sample_decoding,
                                                                     self._lr : lr})
        return loss, accuracy, grad_norm
        
    def val_step(self, sess, enc_inputs, dec_inputs, enc_lens, dec_lens, targets, target_lens, sample_decoding=False):
        batch_size = len(enc_inputs)
        run_vars = [self._loss, self._accuracy]
        
        loss, accuracy = sess.run(run_vars, feed_dict={self._enc_inputs : enc_inputs,
                                                       self._dec_inputs : dec_inputs,
                                                       self._enc_lens : enc_lens,
                                                       self._dec_lens : dec_lens,
                                                       self._targets : targets,
                                                       self._target_lens : target_lens,
                                                       self._dropout_keep_prob : 1.0,
                                                       self._teacher_forcing : False,
                                                       self._teacher_forcing_mask : [1 for _ in range(len(targets[0]))],
                                                       self._batch_size : batch_size,
                                                       self._loss_mask_len : [len(targets[0])],
                                                       self._is_training : False,
                                                       self._sample_decoding : sample_decoding})
        return loss, accuracy
        
    def deploy(self, sess, enc_inputs, enc_lens, dummy_dec_inputs, sample_decoding=False):
        batch_size = len(enc_inputs)
        run_vars = self._predictions
        
        predictions = sess.run(run_vars, feed_dict={self._enc_inputs : enc_inputs,
                                                    self._dec_inputs : dummy_dec_inputs,
                                                    self._enc_lens : enc_lens,
                                                    self._dropout_keep_prob : 1.0,
                                                    self._teacher_forcing : False,
                                                    self._teacher_forcing_mask : [1 for _ in range(len(dummy_dec_inputs[0]))],
                                                    self._batch_size : batch_size, 
                                                    self._is_training : False,
                                                    self._sample_decoding : sample_decoding})
        return predictions