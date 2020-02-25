import tensorflow as tf
import os
import numpy as np
import pickle as pc

debug = False

class Encoder(tf.keras.layers.Layer):
    def __init__(self, batch_size, max_len, enc_unit, dropout_rate, vocab_size, embed_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        
        self.max_len = max_len
        self.batch_size = batch_size
        self.enc_unit = enc_unit
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embed_dim, mask_zero=True)
        self.gru = tf.keras.layers.GRU(self.enc_unit, return_sequences=True, return_state=True, dropout=dropout_rate, recurrent_initializer='glorot_uniform')
        
    def call(self, inp, training):
        # embedding
        embed_input = self.embedding(inp)
        assert embed_input.shape == (self.batch_size, self.max_len, self.embed_dim)

        outputs, state = self.gru(embed_input, training=training, initial_state=tf.zeros((self.batch_size, self.enc_unit)))
        assert outputs.shape == (self.batch_size, self.max_len, self.enc_unit)
        assert state.shape == (self.batch_size, self.enc_unit) 
            
        return outputs, state
    
    def get_config(self):
        config = super(Encoder, self).get_config()
        update = {'hidden_unit': self.enc_unit, 
                  'embed_dim': self.embed_dim,
                  'vocab_size': self.vocab_size}
        
        config.update(update)
        return config