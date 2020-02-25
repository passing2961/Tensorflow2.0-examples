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
from pynori.korean_analyzer import KoreanAnalyzer

np.random.seed(1234)
tf.random.set_seed(1234)

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
WORD_DICT_DIR = 'data/word2idx'

spacer = ChatSpace()

'''nori = KoreanAnalyzer(decompound_mode='DISCARD',
                      discard_punctuation=False,
                      output_unknown_unigrams=False,
                      pos_filter=False,
                      stop_tags=['JKS', 'JKB', 'VV', 'EF'])
'''
def enc_encode(sent, tokenizer):
    return tokenizer.encode(sent) + [tokenizer.vocab_size]

def dec_encode(sent, tokenizer):
    return [tokenizer.vocab_size] + tokenizer.encode(sent) + [tokenizer.vocab_size + 1]

def load_tokenizer(name, corpus=None, target_vocab_size=2**13):
    if os.path.exists(f'data/{name}.subwords'):
        tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(f'data/{name}')
    else:
        tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(corpus, target_vocab_size=target_vocab_size)
        tokenizer.save_to_file(name)
    return tokenizer

def batch_dataset(dataset, batch_size, enc_tokenizer, dec_tokenizer, enc_max_len, dec_max_len):
    buffer_size = len(dataset)
    print(buffer_size)
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
    # pair data load
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
    preds = tf.squeeze(preds, axis=0)
    preds = tf.argmax(preds, axis=-1).numpy()

    pred_str = idx2word(preds, tokenizer, tokenizer_type='decoder')
    '''pred_str = list()
    for token in preds:
        pred_str.append(idx2word[token])
        
        if idx2word[token] == '<eos>':
            break
        
    '''    
    pred_tokens = pred_str
    #print(pred_tokens, len(pred_tokens))
    pred_str = ' '.join(pred_str)
    pred_str = spacer.space(pred_str)

    return pred_str, pred_tokens

def idx2word(sentence, tokenizer, tokenizer_type=None):
    if tokenizer_type == 'encoder':
        eos_idx = tokenizer.vocab_size
    else:
        eos_idx = tokenizer.vocab_size + 1
        #print(tokenizer.decode([7898]))
        
    result = list()
    for token in sentence:
        #print(token)
        
        #print(tokenizer.decode([token]))
        # if token is <eos>, stop the prediction
        if token == eos_idx:
            #result.append('<eos>')
            break
            
        if token == 0:
            continue
            
        if token < tokenizer.vocab_size:
            
            print(tokenizer._id_to_subword(token))
            result.append(tokenizer.decode([token]))
            
    return result

def plot_attention(attention_weights, enc_tokens, pred_tokens):
    #attention_plot = np.zeros((enc_max_len+1, dec_max_len-1))
    
    attention_weights = tf.squeeze(attention_weights, axis=0)
    attention_weights = attention_weights[:len(pred_tokens), :len(enc_tokens)]
    attention_plot = attention_weights.numpy()
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention_plot, cmap='viridis')
    
    fontdict = {'fontsize': 14}
    
    ax.set_xticklabels([''] + enc_tokens, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + pred_tokens, fontdict=fontdict)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.show()
    
def plotly_attention(attention_weights, enc_tokens, pred_tokens):
    attention_weights = tf.squeeze(attention_weights, axis=0)
    attention_weights = attention_weights[:len(pred_tokens), :len(enc_tokens)]
    attention_plot = attention_weights.numpy()
    #print(attention_plot.shape, len(pred_tokens), len(enc_tokens))
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
    
def batch_iter(data, batch_size):
    #num_batches_per_epoch = (len(data) - 1) // batch_size + 1
    num_batches_per_epoch = len(data) // batch_size
    data = np.array(data)
    
    for batch_idx in range(num_batches_per_epoch):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(data))
        enc_data = list()
        dec_data = list()
        
        for each in data[start_idx:end_idx]:
            enc_data.append(each[0])
            dec_data.append(each[1])

        yield enc_data, dec_data
        
def batch_dataset2(batch_x, batch_y, word2idx):
    # batch input & target
    batch_x = list(map(lambda x: x[:enc_max_len], batch_x))
    batch_y = list(map(lambda x: x[:dec_max_len], batch_y))

    batch_x = list(map(lambda x: [word2idx.get(each, word2idx['<unk>']) for each in x], batch_x))
    batch_y = list(map(lambda x: [word2idx.get(each, word2idx['<unk>']) for each in x], batch_y))
                        
    batch_enc_input = list(map(lambda x: list(x) + [word2idx['<eos>']], batch_x))            
    batch_dec_target = list(map(lambda x: [word2idx['<sos>']] + list(x) + [word2idx['<eos>']], batch_y))
            
    batch_enc_input = list(map(lambda x: list(x) + (enc_max_len+1 - len(x)) * [word2idx['<pad>']], batch_enc_input))         
    batch_dec_target = list(map(lambda x: list(x) + (dec_max_len+2 - len(x)) * [word2idx['<pad>']], batch_dec_target))
    
    batch_enc_input = np.array(batch_enc_input)
    batch_dec_target = np.array(batch_dec_target)
    
    return batch_enc_input, batch_dec_target

def build_dict(data, threshold=40000):
    
    if not os.path.exists(WORD_DICT_DIR):
        """
        Build word dictionary
        """
        
        vocab = list()
        for doc in data:
            for word in doc[0]:
                vocab.append(word)
            for word in doc[1]:
                vocab.append(word)
        
        counter = collections.Counter(vocab).most_common(threshold)
        
        word2idx = dict()
        word2idx['<pad>'] = 0
        word2idx['<unk>'] = 1
        word2idx['<sos>'] = 2
        word2idx['<eos>'] = 3
        
        for word, _ in counter:
            word2idx[word] = len(word2idx)
        
        with open(WORD_DICT_DIR, 'wb') as f:
            pc.dump(word2idx, f)
    else:
        """
        Load word dictionary which was built before
        """
        with open(WORD_DICT_DIR, 'rb') as f:
            word2idx = pc.load(f)
    
    print("Load word dictionary and vocab")
    return word2idx

# 형태소 분석기를 통해 나온 결과 filtering
def filtering(result):
    text = result['termAtt']
    morp = result['posTagAtt']
    assert len(text) == len(morp)
    
    temp_list = list()
    for i in range(len(text)):
        if text[i] == ' ':
            continue

        temp_str = text[i] + '/' + morp[i]
        temp_list.append(temp_str)
    return temp_list
    
def load_morp_dataset(pair_data):
    # 형태소 분석된 [질문, 응답] 데이터셋 구축
    total_data = list()
    for each in pair_data:
        utter_result = nori.do_analysis(each[0])
        resp_result = nori.do_analysis(each[1])

        utter = filtering(utter_result)
        resp = filtering(resp_result)

        total_data.append([utter, resp])

    # 데이터 사이즈 및 실제 결과 확인
    print("Total size of data is", len(total_data))
    print("\nExample:")
    print(total_data[:1])
    
    return total_data
    
def test_batch_dataset(batch_x, word2idx):
    # batch input
    batch_x = list(map(lambda x: x[:enc_max_len], batch_x))

    batch_x = list(map(lambda x: [word2idx.get(each, word2idx['<unk>']) for each in x], batch_x))
                        
    batch_enc_input = list(map(lambda x: list(x) + [word2idx['<eos>']], batch_x))
            
    batch_enc_input = list(map(lambda x: list(x) + (enc_max_len+1 - len(x)) * [word2idx['<pad>']], batch_enc_input))
    
    batch_enc_input = np.array(batch_enc_input)
    
    return batch_enc_input