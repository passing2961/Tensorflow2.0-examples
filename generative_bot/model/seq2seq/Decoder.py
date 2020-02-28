import tensorflow as tf

class Decoder(tf.keras.layers.Layer):
    def __init__(self, batch_size, max_len, dec_unit, embed_dim, vocab_size, dropout_rate):
        super(Decoder, self).__init__()
        
        # Define the hyper-parameters
        self.batch_size = batch_size
        self.dec_unit = dec_unit
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.dropout_rate = dropout_rate
        
        # Define embedding & gru layers
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embed_dim)
        self.gru = tf.keras.layers.GRU(self.dec_unit, return_sequences=True, return_state=True, dropout=self.dropout_rate, recurrent_initializer='glorot_uniform')
        
        # Define linear layers
        self.fc = tf.keras.layers.Dense(self.vocab_size)
        
    def call(self, dec_input, dec_hidden, training):
        # Convert word index to embeddings (mapping discrete tokens to continuous space)
        # embed_input shape == (batch_size, 1, embed_dim)
        embed_input = self.embedding(dec_input)
        
        # Forward through GRU module
        # outputs shape == (batch_size, 1, dec_unit)
        # state shape == (batch_size, dec_unit)
        outputs, state = self.gru(embed_input, dec_hidden, training=training)
        
        # Predict next word
        # logits shape == (batch_size, 1, vocab_size)
        logits = self.fc(outputs)
        
        return logits, state