import os
import pickle as pc
import csv
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.graph_objs as go
from plotly.offline import iplot
import plotly.figure_factory as ff
import collections

from chatspace import ChatSpace

# For the reproducibility
np.random.seed(1234)
tf.random.set_seed(1234)

# Define the hyper-parameters and arguments
lr = 1e-3
batch_size = 32
enc_max_len = 25
dec_max_len = 25
enc_unit = 300
dec_unit = 300
embed_dim = 200
dropout_rate = 0.3
epochs = 20
log_interval = 50
data_dir = 'data/ChatbotData.csv'

# Initialize the chatspace module
spacer = ChatSpace()

def enc_encode(sent, tokenizer):
    '''
    Mapping the encoder's input sentence into indexes by using the tokenizer
        - tokenizer.vocab_size => index of <eos> token
    '''
    return tokenizer.encode(sent) + [tokenizer.vocab_size]

def dec_encode(sent, tokenizer):
    '''
    Mapping the decoder's input sentence into indexes by using the tokenizer
        - tokenizer.vocab_size => index of <sos> token
        - tokenizer.vocab_size+1 => index of <eos> token
    '''
    return [tokenizer.vocab_size] + tokenizer.encode(sent) + [tokenizer.vocab_size + 1]

def load_tokenizer(name, corpus=None, target_vocab_size=2**13):
    '''
    Create a custom subwords tokenizer from the training dataset
    '''
    if os.path.exists(f'data/{name}.subwords'):
        tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(f'data/{name}')
    else:
        tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(corpus, target_vocab_size=target_vocab_size)
        tokenizer.save_to_file(name)
    return tokenizer

def batch_dataset(dataset, batch_size, enc_tokenizer, dec_tokenizer, enc_max_len, dec_max_len):
    '''
    Prepare the mini-batch dataset for training
        1) Padding & Truncating with the maximum length
        2) Shuffling for keeping the principle of stochastic gradient descent
    Cache the dataset to memory to get a speedup while reading from it
    '''
    buffer_size = len(dataset)

    pad_x = tf.keras.preprocessing.sequence.pad_sequences([enc_encode(x, enc_tokenizer) for x, y in dataset], 
                                                             maxlen=enc_max_len+1, padding='post')
    pad_y = tf.keras.preprocessing.sequence.pad_sequences([dec_encode(y, dec_tokenizer) for x, y in dataset], 
                                                             maxlen=dec_max_len+2, padding='post')
    assert len(pad_x) == len(pad_y)
    
    dataset_tensor = tf.data.Dataset.from_tensor_slices((pad_x, pad_y))
    dataset_tensor = dataset_tensor.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    dataset_tensor = dataset_tensor.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset_tensor

def load_dataset(data_dir):
    '''
    Load the chatbot dataset
    Reference: https://github.com/songys/Chatbot_data
    '''
    pair_data = list()

    f = open(data_dir, 'r', encoding='utf-8')
    reader = csv.reader(f)
    for idx, line in enumerate(reader):
        if idx == 0:
            continue

        pair_data.append([line[0], line[1]])
    f.close()
    
    return pair_data

def decoding_from_result(preds, tokenizer):
    '''
    Decode the predicted token indexes into the word tokens
        1) idx2word: Iteratively, decode one word token at a time
        2) spacer.space: Spacing model that fits well with chat by PingPong
            Reference: https://github.com/pingpong-ai/chatspace
    '''
    preds = tf.squeeze(preds, axis=0)
    preds = tf.argmax(preds, axis=-1).numpy()

    pred_str = idx2word(preds, tokenizer, tokenizer_type='decoder')

    pred_tokens = pred_str
    
    pred_str = ' '.join(pred_str)
    pred_str = spacer.space(pred_str)

    return pred_str, pred_tokens

def idx2word(sentence, tokenizer, tokenizer_type=None):
    '''
    Iteratively, decode one word token at a time
    '''
    
    if tokenizer_type == 'encoder':
        eos_idx = tokenizer.vocab_size
    else:
        eos_idx = tokenizer.vocab_size + 1
        
    result = list()
    for token in sentence:
        # if token is <eos>, stop the prediction
        if token == eos_idx:
            break
            
        if token == 0:
            continue
            
        if token < tokenizer.vocab_size:
            result.append(tokenizer.decode([token]))
            
    return result

  
def plotly_attention(attention_weights, enc_tokens, pred_tokens):
    '''
    Visualize the attention weights by using plotly
    (Note: We had a korean encoding error related with matplotlib, so we used plotly) 
    '''
    attention_weights = tf.squeeze(attention_weights, axis=0)
    attention_weights = attention_weights[:len(pred_tokens), :len(enc_tokens)]
    attention_plot = attention_weights.numpy()
    
    layout_heatmap = go.Layout(
        title=('Attention'),
        xaxis=dict(),
        yaxis=dict()
    )
    
    ff_fig = ff.create_annotated_heatmap(x=enc_tokens, y=pred_tokens, z=attention_plot, colorscale = 'Viridis')
    
    fig = go.FigureWidget(ff_fig)
    fig.layout = layout_heatmap
    fig.layout.annotations = ff_fig.layout.annotations
    fig.data[0].colorbar = dict(title='attention weights', titleside='right')
    
    iplot(fig)