import tensorflow as tf
from model.seq2seq_attn.Attention import *

class Decoder(tf.keras.layers.Layer):
    def __init__(self, attn_type, method, batch_size, max_len, dec_unit, embed_dim, vocab_size, dropout_rate, **kwargs):
        super(Decoder, self).__init__()
        
        self.batch_size = batch_size
        self.dec_unit = dec_unit
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.attn_type = attn_type
        self.method = method
        
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embed_dim)
        self.gru = tf.keras.layers.GRU(self.dec_unit, return_sequences=True, return_state=True, dropout=dropout_rate, recurrent_initializer='glorot_uniform')
        
        self.fc = tf.keras.layers.Dense(self.vocab_size)
        
        if self.attn_type == 'bahdanau':
            self.attn = BahdanauAttention(self.dec_unit)
        elif self.attn_type == 'luong':
            self.attn = LuongAttention(self.method, self.dec_unit)
        else:
            raise ValueError(self.attn_type, "is not an appropriate attention mechanism.")
    
    def call(self, dec_input, dec_hidden, enc_output, training=True, source=None):
        # embed_input shape == (batch_size, dec_max_len, embed_dim)
        embed_input = self.embedding(dec_input)
        
        # context_vector shape == (batch_size, enc_unit)
        # attention_weights shape == (batch_size, enc_max_len, 1)
        if self.attn_type == 'bahdanau':
            logits, state, attention_weights = self.bahdanau_process(embed_input, dec_hidden, enc_output, source, training)
        elif self.attn_type == 'luong':
            logits, state, attention_weights = self.luong_process(embed_input, dec_hidden, enc_output, source, training)
            
        return logits, state, attention_weights
    
       
    def bahdanau_process(self, embed_input, dec_hidden, enc_output, source, training):
        # context_vector shape == (batch_size, enc_unit)
        # attention_weights shape == (batch_size, enc_max_len, 1)
        context_vector, attention_weights = self.attn(dec_hidden, enc_output, source)

        # concat_input shape == (batch_size, 1, enc_unit + embed_dim)
        concat_input = tf.concat([tf.expand_dims(context_vector, 1), embed_input], axis=-1)

        # outputs shape == (batch_size, 1, dec_unit)
        # state shape == (batch_size, dec_unit)
        outputs, state = self.gru(concat_input, dec_hidden, training=training)
        
        logits = self.fc(outputs)
        
        return logits, state, attention_weights
    
    def luong_process(self, embed_input, dec_hidden, enc_output, source, training):
        # outputs shape == (batch_size, 1, dec_unit)
        # state shape == (batch_size, dec_unit)
        outputs, state = self.gru(embed_input, dec_hidden, training=training)
        
        # context_vector shape == (batch_size, enc_unit)
        # attention_weights shape == (batch_size, enc_max_len, 1)
        context_vector, attention_weights = self.attn(state, enc_output, source)
        
        concat_input = tf.concat([tf.expand_dims(context_vector, 1), outputs], axis=-1)
        concat_input = tf.nn.tanh(concat_input)
        
        logits = self.fc(concat_input)
        
        return logits, state, attention_weights