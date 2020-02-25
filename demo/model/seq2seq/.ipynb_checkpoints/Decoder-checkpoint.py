import tensorflow as tf
#from Attention import *

class Decoder(tf.keras.layers.Layer):
    def __init__(self, batch_size, max_len, dec_unit, embed_dim, vocab_size, dropout_rate, **kwargs):
        super(Decoder, self).__init__()
        
        self.batch_size = batch_size
        self.dec_unit = dec_unit
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embed_dim)
        self.gru = tf.keras.layers.GRU(self.dec_unit, return_sequences=True, return_state=True, dropout=dropout_rate, recurrent_initializer='glorot_uniform')
        
        self.fc = tf.keras.layers.Dense(self.vocab_size)
        
        #self.attn = LuongAttention(self.dec_unit)
        
    def call(self, dec_input, dec_hidden, training):
        embed_input = self.embedding(dec_input)
        
        #context_vector, attn_weights = self.attn(dec_hidden, enc_output)
        
        outputs, state = self.gru(embed_input, dec_hidden, training=training)
        
        #outputs = tf.reshape(outputs, (-1, outputs.shape[2]))
        
        logits = self.fc(outputs)
        
        return logits, state