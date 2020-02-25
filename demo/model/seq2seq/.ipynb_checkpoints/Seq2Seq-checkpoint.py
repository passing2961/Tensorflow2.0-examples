import tensorflow as tf
from model.seq2seq.Encoder import *
from model.seq2seq.Decoder import *

class seq2seq(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(seq2seq, self).__init__()
        
        self.batch_size = config['batch_size']
        self.enc_max_len = config['enc_max_len']
        self.dec_max_len = config['dec_max_len']
        self.enc_unit = config['enc_unit']
        self.dec_unit = config['dec_unit']
        self.embed_dim = config['embed_dim']
        self.dropout_rate = config['dropout_rate']
        self.enc_vocab_size = config['enc_vocab_size']
        self.dec_vocab_size = config['dec_vocab_size']
        self.sos_token = config['dec_sos_token']
        
        self.encoder = Encoder(self.batch_size, self.enc_max_len, self.enc_unit,
                               self.dropout_rate, self.enc_vocab_size, self.embed_dim)
        self.decoder = Decoder(self.batch_size, self.dec_max_len, self.dec_unit,
                               self.embed_dim, self.dec_vocab_size, self.dropout_rate)
        
        
    def call(self, enc_input, dec_target=None, training=True):

        enc_outputs, enc_state = self.encoder(enc_input, training=training)
        
        dec_input = tf.expand_dims([self.sos_token] * self.batch_size, 1)
        dec_hidden = enc_state
        
        seq2seq_output = tf.zeros((self.batch_size, 1, self.dec_vocab_size), dtype=tf.float32)

        if dec_target is None:
            dec_time_steps = self.dec_max_len
        else:
            dec_time_steps = dec_target.shape[1]
            
        for t in range(1, dec_time_steps):
            
            dec_outputs, dec_hidden = self.decoder(dec_input, dec_hidden, training=training)
            
            seq2seq_output = tf.concat([seq2seq_output, dec_outputs[:, :1]], axis=1)
            
            if dec_target is None:
                dec_input = tf.argmax(dec_outputs, axis=-1, output_type=tf.int32)
            else:
                dec_input = tf.expand_dims(dec_target[:,t], 1)
        
        return seq2seq_output[:, 1:]