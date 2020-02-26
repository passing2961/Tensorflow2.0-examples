import tensorflow as tf
from model.seq2seq_attn.Encoder import *
from model.seq2seq_attn.Decoder import *

class seq2seq_attn(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(seq2seq_attn, self).__init__()
        
        # Define the hyper-parameters or arguments
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
        self.attn_type = config['attn_type']
        self.method = config['method']
        
        # Define the encoder decoder layer
        self.encoder = Encoder(self.batch_size, self.enc_max_len, self.enc_unit,
                               self.dropout_rate, self.enc_vocab_size, self.embed_dim)
        self.decoder = Decoder(self.attn_type, self.method, self.batch_size, self.dec_max_len, self.dec_unit,
                               self.embed_dim, self.dec_vocab_size, self.dropout_rate)
        
        
    def call(self, enc_input, dec_target=None, training=True):
        source = enc_input
        
        # Feed the enc_input into the encoder
        enc_outputs, enc_state = self.encoder(enc_input, training=training)
        
        # Create initial decoder input (start with <sos> tokens for each sentence)
        # dec_hidden is initialized at the decoder's gru hidden state at first time step
        dec_input = tf.expand_dims([self.sos_token] * self.batch_size, 1)
        dec_hidden = enc_state
        
        seq2seq_output = tf.zeros((self.batch_size, 1, self.dec_vocab_size), dtype=tf.float32)
        seq2seq_attn_weights = tf.zeros((self.batch_size, 1, self.enc_max_len), dtype=tf.float32)
        
        # In the prediction process, not a training, we don't know how much time we need (= dec_time_steps)
        # since the dec_target is not given at the prediction
        # Thus, if dec_target is None, then we should use the dec_time_steps as self.dec_max_len        
        if dec_target is None:
            dec_time_steps = self.dec_max_len
        else:
            dec_time_steps = dec_target.shape[1]
        
        # use teacher forcing
        for t in range(1, dec_time_steps):
            # Forward the decoder at each time step
            dec_outputs, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, enc_outputs, training=training, source=source)
        
            # This is for calculating loss for after this loop
            # Also, for visualizing attention weights
            # seq2seq_output shape == (batch_size, t+1, dec_vocab_size)
            # seq2seq_attn_weights shape == (batch_size, t+1, enc_max_len)
            seq2seq_output = tf.concat([seq2seq_output, dec_outputs[:, :1]], axis=1)
            seq2seq_attn_weights = tf.concat([seq2seq_attn_weights, tf.einsum('ijk->ikj', attention_weights)], axis=1)
            
            # if dec_target is None, then we will use the previous predicted word token 
            # as the input token at the current time step
            # if dec_target is not None, then we will apply the teacher forcing technique
            if dec_target is None:
                dec_input = tf.argmax(dec_outputs, axis=-1, output_type=tf.int32)
            else:
                dec_input = tf.expand_dims(dec_target[:,t], 1)
        
        # final output
        # seq2seq_output shape == (batch_size, dec_max_len-1, dec_vocab_size)
        # seq2seq_attn_weights shape == (batch_size, dec_max_len-1, enc_max_len)
        return seq2seq_output[:, 1:], seq2seq_attn_weights[:, 1:]