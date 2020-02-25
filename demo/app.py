import os
import csv
import json
import numpy as np
import tensorflow as tf

from model.seq2seq.Seq2Seq import *
from utils.utils import *
from flask import Flask, request, jsonify

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/chatbot_demo', methods=['GET', 'POST'])
def showDemo():
    
    # Load tokenizer
    enc_tokenizer = load_tokenizer('enc-tokenizer')
    dec_tokenizer = load_tokenizer('dec-tokenizer')
    enc_vocab_size = enc_tokenizer.vocab_size + 1
    dec_vocab_size = dec_tokenizer.vocab_size + 2
    
    # Define the optimizer and the loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    # Define seq2seq model
    config = {'batch_size': 1,
              'enc_max_len': enc_max_len+1,
              'dec_max_len': dec_max_len+2,
              'enc_unit': enc_unit,
              'dec_unit': dec_unit,
              'embed_dim': embed_dim,
              'dropout_rate': dropout_rate,
              'enc_vocab_size': enc_vocab_size,
              'dec_vocab_size': dec_vocab_size,
              'dec_sos_token': dec_tokenizer.vocab_size}

    model = seq2seq(config)
        
    # checkpoint
    checkpoint_dir = 'checkpoint/daily-korean'
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    
    if request.method == 'GET':
        print("GET")
    elif request.method == 'POST':
        input_text = request.form['text']        
        
        enc_input = tf.keras.preprocessing.sequence.pad_sequences([enc_encode(input_text, enc_tokenizer)], 
                                                             maxlen=enc_max_len+1, padding='post')
        
        
        model.load_weights(filepath=tf.train.latest_checkpoint(checkpoint_dir))

        preds = model(enc_input, training=False)
        
        pred_str, pred_tokens = decoding_from_result(preds, dec_tokenizer)
        
        return jsonify(response=pred_str)
    
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port='6006')