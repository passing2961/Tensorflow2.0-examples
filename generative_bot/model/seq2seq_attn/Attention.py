import tensorflow as tf

class BahdanauAttention(tf.keras.layers.Layer):
    '''
    Reference:
     "Neural Machine Translation by Jointly Learning to Align and Translate"
     https://arxiv.org/abs/1409.0473
    '''
    def __init__(self, attn_unit, **kwargs):
        super(BahdanauAttention, self).__init__()
        
        self.attn_unit = attn_unit
        
        # Define several linear layer
        self.W1 = tf.keras.layers.Dense(self.attn_unit)
        self.W2 = tf.keras.layers.Dense(self.attn_unit)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, query, values, source=None):
        # query: decoder hidden state at previous time step t-1
        # values: encoder outputs
        # source: encoder inputs which are represented with indexes (for calculating masked softmax)
        # query shape == (batch_size, dec_unit)
        # values shape == (batch_size, enc_max_len, enc_unit)
        # source shape == (batch_size, enc_max_len)
        
        # query_hidden shape == (batch_size, 1, dec_unit)
        # score shape == (batch_size, enc_max_len, 1)
        query_hidden = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(self.W1(query_hidden) + self.W2(values)))
        
        # masking attention score corresponding to pad values in encoder inputs
        # attention_weights shape == (batch_size, enc_max_len, 1)
        masked_score = self.masking_attn_score(source, score)
        masked_score = masked_score[:, :, tf.newaxis]
        attention_weights = tf.nn.softmax(masked_score, axis=1)

        # context_vector (before sum) shape == (batch_size, enc_max_len, enc_unit)
        # context_vector (after sum) shape == (batch_size, enc_unit)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights
    
    def masking_attn_score(self, source, score):
        # The mask is multiplied with -1e9 (close to negative infinity)
        # The large negative inputs to softmax are near zero in the output
        # because the output of softmax function is normalized into 0~1
        # source shape == (batch_size, enc_max_len)
        # score shape == (batch_size, enc_max_len)        
        score = tf.squeeze(score, axis=-1)
        mask = tf.math.logical_not(tf.math.not_equal(source, 0))
        mask = tf.cast(mask, dtype=score.dtype)
        mask *= -1e9

        return score + mask
        
class LuongAttention(tf.keras.layers.Layer):
    '''
    Reference:
     "Effective approaches to Attention-based Neural Machine Translation"
     https://arxiv.org/abs/1508.04025
    '''
    def __init__(self, method, attn_unit, **kwargs):
        super(LuongAttention, self).__init__()
        
        # method: indicator that determines the score function
        self.method = method
        self.attn_unit = attn_unit
        
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention score function.")
        
    def dot_score(self, query, values):
        # [before reduce_sum] shape == (batch_size, enc_max_len, enc_unit)
        # [after reduce_sum] shape == (batch_size, enc_max_len)
        return tf.reduce_sum(query * values, axis=2)
        
    def call(self, query, values, source=None):
        # query: decoder hidden state at current time step t
        # values: encoder outputs
        # source: encoder inputs which are represented with indexes (for calculating masked softmax)
        # query shape == (batch_size, dec_unit)
        # values shape == (batch_size, enc_max_len, enc_unit)
        # source shape == (batch_size, enc_max_len)
        
        # query_hidden shape == (batch_size, 1, dec_unit)
        # score shape == (batch_size, enc_max_len, 1)
        query_hidden = tf.expand_dims(query, 1)

        if self.method == 'dot':
            attn_score = self.dot_score(query_hidden, values)
        
        # masking attention score corresponding to pad values in encoder inputs
        # attention_weights shape == (batch_size, enc_max_len, 1)
        masked_score = self.masking_attn_score(source, attn_score)
        masked_score = masked_score[:, :, tf.newaxis]
        attention_weights = tf.nn.softmax(masked_score, axis=1)

        # context_vector (before sum) shape == (batch_size, enc_max_len, enc_unit)
        # context_vector (after sum) shape == (batch_size, enc_unit)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights
        
    def masking_attn_score(self, source, score):
        # The mask is multiplied with -1e9 (close to negative infinity)
        # The large negative inputs to softmax are near zero in the output
        # because the output of softmax function is normalized into 0~1
        # source shape == (batch_size, enc_max_len)
        # score shape == (batch_size, enc_max_len)
        mask = tf.math.logical_not(tf.math.not_equal(source, 0))
        mask = tf.cast(mask, dtype=score.dtype)
        mask *= -1e9

        return score + mask
        