# Tensorflow2.0-examples
Implementations of deep learning model with Tensorflow 2.0

## Overview

This repository is specialized for neural response generation or dialogue generation in the field of NLP.

## Models

This repository consists of various models related with dialogue generation:
  
  1. Sequence-to-Sequence Model
  2. Sequence-to-Sequence Model with Attention Mechanisms (Bahdanau & Luong)
  3. Hierarchical Recurrent Encoder-Decoder (HRED) *-> Not yet*
  4. Latent Variable Hierarchical Recurrent Encoder-Decoder (VHRED) *-> Not yet*
  5. Transformer *-> Not yet*

## To Do

[DailyDialog]: https://arxiv.org/abs/1710.03957

- Applying the earlystopping technique
- Applying the beam search
- Tokenizing sentence at morepheme level
- Use other datasets (e.g. [DailyDialog])

## Datasets

[link]: https://github.com/songys/Chatbot_data

To train and evaluate several models, we used korean daily chat data.
You can be publicly available to download dataset by following this [link].

## Requirements

- Install `tensorflow 2.0`

```
$ pip install tensorflow
```
- Install `ChatSpace`
  
```
$ pip install chatspace
```

## Reference

[Tensorflow 2.0 Neural machine translation with attention]: https://www.tensorflow.org/tutorials/text/nmt_with_attention
[Tensorflow 2.0 Encoder-Decoder]: https://github.com/yusugomori/deeplearning-tf2
[ChatSpace]: https://github.com/pingpong-ai/chatspace

- [Tensorflow 2.0 Neural machine translation with attention]
- [Tensorflow 2.0 Encoder-Decoder]
- [ChatSpace]











