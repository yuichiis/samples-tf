# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
from numpy import ndarray
import os
import io
import time

# Download the file
class EngFraDataset:
    def download(self):
        path_to_zip = tf.keras.utils.get_file(
        'fra-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip',
        extract=True)

        path_to_file = os.path.dirname(path_to_zip)+"/fra.txt"
        return path_to_file

    # Converts the unicode file to ascii
    def unicode_to_ascii(self,s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

    def preprocess_sentence(self,w):
        w = self.unicode_to_ascii(w.lower().strip())
        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
        w = w.strip()
        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        w = '<start> ' + w + ' <end>'
        return w

    # 1. Remove the accents
    # 2. Clean the sentences
    # 3. Return word pairs in the format: [ENGLISH, SPANISH]
    def create_dataset(self, path, num_examples):
        lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        word_pairs = [[self.preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
        return zip(*word_pairs)

    def tokenize(self, lang, num_words=None):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=num_words, filters='')
        lang_tokenizer.fit_on_texts(lang)
        tensor = lang_tokenizer.texts_to_sequences(lang)
        tensor = tf.keras.utils.pad_sequences(tensor,
                                                         padding='post')
        return tensor, lang_tokenizer

    def load_data(self, path=None, num_examples=None, num_words=None):
        if path is None:
            path = self.download()
        # creating cleaned input, output pairs
        targ_lang, inp_lang = self.create_dataset(path, num_examples)

        input_tensor, inp_lang_tokenizer = self.tokenize(inp_lang,num_words=num_words)
        target_tensor, targ_lang_tokenizer = self.tokenize(targ_lang,num_words=num_words)
        choice = np.random.choice(len(input_tensor),len(input_tensor),replace=False)
        input_tensor = self.shuffle(input_tensor,choice)
        target_tensor = self.shuffle(target_tensor,choice)

        return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

    def shuffle(self, tensor, choice):
        result = np.zeros_like(tensor)
        for i in range(len(tensor)):
            result[i,:] = tensor[choice[i],:]
        return result

    def convert(self, lang, tensor):
        for t in tensor:
            if t!=0:
                print ("%d ----> %s" % (t, lang.index_word[t]))

    def seq2str(self,sequence,lang):
        result = ''
        for word_id in sequence:
            if word_id == 0:
                result += ' '
            else:
                word = lang.index_word[word_id]
                if word == '<end>':
                    return result
                if word != '<start>':
                    result += word + ' '
        return result

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

#n, d = 2048, 512
#pos_encoding = positional_encoding(n, d)
#print(pos_encoding.shape)
#pos_encoding = pos_encoding[0]
#
## Juggle the dimensions for the plot
#pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2))
#pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
#pos_encoding = tf.reshape(pos_encoding, (d, n))
#
#plt.pcolormesh(pos_encoding, cmap='RdBu')
#plt.ylabel('Depth')
#plt.xlabel('Position')
#plt.colorbar()
#plt.show()

def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32) # 0:pass 1:masking

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

#x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
#create_padding_mask(x)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0) # 0:pass 1:masking
  return mask  # (seq_len, seq_len)

#x = tf.random.uniform((1, 3))
#temp = create_look_ahead_mask(x.shape[1])
#temp

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)
    #print('score:',scaled_attention_logits.shape)
    #print('mask:',mask.shape)
    #print('score:',scaled_attention_logits)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

def split_heads(x, batch_size=1,num_heads=1,depth=1):
  """Split the last dimension into (num_heads, depth).
  Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
  """
  x = tf.reshape(x, (batch_size, -1, num_heads, depth))
  return tf.transpose(x, perm=[0, 2, 1, 3])

def print_out_attention(q, k, v, mask=None):
  temp_out, temp_attn = scaled_dot_product_attention(
      q, k, v, mask)
  print('Attention weights are:')
  print(temp_attn)
  print('Output is:')
  print(temp_out)

np.set_printoptions(suppress=True)

#temp_k = tf.constant([[10, 0, 0],
#                      [0, 10, 0],
#                      [0, 0, 10],
#                      [0, 0, 10]], dtype=tf.float32)  # (4, 3)
#
#temp_v = tf.constant([[1, 0],
#                      [10, 0],
#                      [100, 5],
#                      [1000, 6]], dtype=tf.float32)  # (4, 2)
#
## This `query` aligns with the second `key`,
## so the second `value` is returned.

#batch_size = 2
#num_heads = 1
#dimsize = 2
#depth = dimsize // num_heads
#print('depth:',depth)
#temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
#temp_q = tf.constant([[[1,1],[1,1],[1,1]],[[1,1],[1,1],[1,1]]],dtype=tf.float32)
#temp_v = tf.constant([[[1,1],[1,1],[1,1],[1,1]],[[1,1],[1,1],[1,1],[1,1]]],dtype=tf.float32)
#temp_k = tf.constant([[[1,1],[1,1],[1,1],[1,1]],[[1,1],[1,1],[1,1],[1,1]]],dtype=tf.float32)
#seq = tf.constant([[1,2,3],[1,2,3]])
#mask = create_padding_mask(seq)
#temp_q = split_heads(temp_q,batch_size,num_heads,depth)
#temp_v = split_heads(temp_v,batch_size,num_heads,depth)
#temp_k = split_heads(temp_k,batch_size,num_heads,depth)
#print(temp_q.shape)
#print(temp_v.shape)
#print(temp_k.shape)
#print(mask.shape)
#print_out_attention(temp_q, temp_k, temp_v, mask)
#exit()
# This query aligns with a repeated key (third and fourth),
# so all associated values get averaged.
#temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
#print_out(temp_q, temp_k, temp_v)

#temp_q = tf.constant([[0, 0, 10],
#                      [0, 10, 0],
#                      [10, 10, 0]], dtype=tf.float32)  # (3, 3)
#print_out(temp_q, temp_k, temp_v)


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights
  
#temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
#y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
#out, attn = temp_mha(y, k=y, q=y, mask=None)
#out.shape, attn.shape

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

#sample_ffn = point_wise_feed_forward_network(512, 2048)
#sample_ffn(tf.random.uniform((64, 50, 512))).shape

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2

#sample_encoder_layer = EncoderLayer(512, 8, 2048)
#
#sample_encoder_layer_output = sample_encoder_layer(
#    tf.random.uniform((64, 43, 512)), False, None)
#
#sample_encoder_layer_output.shape  # (batch_size, input_seq_len, d_model)

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)
    #print('===============================')
    #print('x:',x.shape)
    #print('look_ahead_mask:',look_ahead_mask.shape)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

    return out3, attn_weights_block1, attn_weights_block2
  
#sample_decoder_layer = DecoderLayer(512, 8, 2048)
#
#sample_decoder_layer_output, _, _ = sample_decoder_layer(
#    tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
#    False, None, None)
#
#sample_decoder_layer_output.shape  # (batch_size, target_seq_len, d_model)

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding,
                                            self.d_model)

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]

    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    return x  # (batch_size, input_seq_len, d_model)

#sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
#                         dff=2048, input_vocab_size=8500,
#                         maximum_position_encoding=10000)
#temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)
#
#sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)
#
#print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)

class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}

    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)

      attention_weights[f'decoder_layer{i+1}_block1'] = block1
      attention_weights[f'decoder_layer{i+1}_block2'] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights

#sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
#                         dff=2048, target_vocab_size=8000,
#                         maximum_position_encoding=5000)
#temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)
#
#output, attn = sample_decoder(temp_input,
#                              enc_output=sample_encoder_output,
#                              training=False,
#                              look_ahead_mask=None,
#                              padding_mask=None)
#
#output.shape, attn['decoder_layer2_block2'].shape

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, pe_input, pe_target, rate=0.1):
    super().__init__()
    self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                             input_vocab_size, pe_input, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                           target_vocab_size, pe_target, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs, training):
    # Keras models prefer if you pass all your inputs in the first argument
    inp, tar = inputs

    enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)

    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, enc_output, training, look_ahead_mask, dec_padding_mask)

    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return final_output, attention_weights

  def create_masks(self, inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)  # 0:pass 1:masking

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)  # 0:pass 1:masking

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    #print('===================')
    #print(look_ahead_mask.shape)
    #print(dec_target_padding_mask.shape)
    look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask) # padding or ahead
    #print(look_ahead_mask.shape)

    return enc_padding_mask, look_ahead_mask, dec_padding_mask

#sample_transformer = Transformer(
#    num_layers=2, d_model=512, num_heads=8, dff=2048,
#    input_vocab_size=8500, target_vocab_size=8000,
#    pe_input=10000, pe_target=6000)
#
#temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
#temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)
#
#fn_out, _ = sample_transformer([temp_input, temp_target], training=False)
#
#fn_out.shape  # (batch_size, tar_seq_len, target_vocab_size)

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

num_examples = 20000 #10000 #30000
num_words = 1024 #None #128

EPOCHS = 10#20
BATCH_SIZE = 64



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  
learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

#temp_learning_rate_schedule = CustomSchedule(d_model)
#
#plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
#plt.ylabel("Learning Rate")
#plt.xlabel("Train Step")

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')


# Try experimenting with the size of that dataset
dataset = EngFraDataset()
path_to_file = dataset.download()
print("Generating data...")
input_tensor, target_tensor, inp_lang, targ_lang = dataset.load_data(path_to_file, num_examples,num_words=num_words)
input_vocab_size = len(inp_lang.index_word)+1
target_vocab_size = len(targ_lang.index_word)+1
if num_words is not None:
    input_vocab_size = min(input_vocab_size ,num_words)
    target_vocab_size = min(target_vocab_size,num_words)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

# Creating training and validation sets using an 80-20 split
#input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)
split_at = len(input_tensor) - len(input_tensor) // 10
input_tensor_train,  input_tensor_val  = (input_tensor[:split_at], input_tensor[split_at:])
target_tensor_train, target_tensor_val = (target_tensor[:split_at],target_tensor[split_at:])

def make_labels(label_tensor):
  lebel_len,lebel_words = label_tensor.shape
  label_tensor  = label_tensor[:,1:lebel_words]
  filler = np.zeros(lebel_len,dtype=label_tensor.dtype).reshape(lebel_len,1)
  label_tensor  = np.append(label_tensor,filler,axis=1)
  return label_tensor

label_tensor_train = make_labels(target_tensor_train)
label_tensor_val   = make_labels(target_tensor_val)


# Show length
#print('input=',input_tensor_train.shape)
#print('target=',target_tensor_train.shape)
#print('label=',label_tensor_train.shape)
#print('val_input=',input_tensor_val.shape)
#print('val_target=', target_tensor_val.shape)
#
#print('input=',input_tensor_train[10])
#print('target=',target_tensor_train[10])
#print('label=',label_tensor_train[10])

BUFFER_SIZE = len(input_tensor_train)
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
print("num_examples:",num_examples)
print('num_words:',num_words)
print("epoch:",EPOCHS)
print("batch_size:",BATCH_SIZE)
print("embedding_dim:",d_model)
print("layers: ",num_layers)
print("units: ",dff)
print("heads: ",num_heads)
print("dropout_rate: ",dropout_rate)
print("Input  length:",max_length_inp)
print("Target length:",max_length_targ)
print("Input  word dictionary: %d(%d)" % (input_vocab_size,len(inp_lang.index_word)+1))
print("Target word dictionary: %d(%d)" % (target_vocab_size,len(targ_lang.index_word)+1))
print(" ")


transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    pe_input=1000,
    pe_target=1000,
    rate=dropout_rate)

#checkpoint_path = "./checkpoints/train"

#ckpt = tf.train.Checkpoint(transformer=transformer,
#                           optimizer=optimizer)

#ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
#if ckpt_manager.latest_checkpoint:
#  ckpt.restore(ckpt_manager.latest_checkpoint)
#  print('Latest checkpoint restored!!')

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

def make_batch(batch,batch_size,input_data,target_data):
  start_pointer = batch*batch_size
  end_pointer = start_pointer+batch_size
  if(start_pointer>=len(input_data)):
    return [None,None,None]
  if(end_pointer>=len(input_data)):
    end_pointer = len(input_data)
  input = input_data[start_pointer:end_pointer]
  target = target_data[start_pointer:end_pointer]
  return [batch+1,input,target]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]

  with tf.GradientTape() as tape:
    predictions, _ = transformer([inp, tar_inp],
                                 training = True)
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_accuracy(accuracy_function(tar_real, predictions))

for epoch in range(EPOCHS):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()

  batch = 0
  # inp -> portuguese, tar -> english
  while True:
    batch, inp, tar = make_batch(batch,BATCH_SIZE,input_tensor_train,target_tensor_train)
    if(batch is None):
       break
    train_step(inp, tar)

    if batch % 50 == 0:
      print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

  #if (epoch + 1) % 5 == 0:
  #  ckpt_save_path = ckpt_manager.save()
  #  print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

  print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

  print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

class Translator(tf.Module):
  def __init__(self, transformer,
      max_out_length=None, start_voc_id=None, end_voc_id=None):
    #self.tokenizers = tokenizers
    self.transformer = transformer
    self.max_out_length = max_out_length
    self.start_voc_id = start_voc_id
    self.end_voc_id = end_voc_id

  def __call__(self, sentence, max_length=20):
    # input sentence is portuguese, hence adding the start and end token
    assert isinstance(sentence, tf.Tensor)
    if len(sentence.shape) == 0:
      sentence = sentence[tf.newaxis]

    #sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

    encoder_input = tf.expand_dims(sentence,axis=0)

    # as the target is english, the first token to the transformer should be the
    # english start token.
    #start_end = self.tokenizers.en.tokenize([''])[0]
    #start = start_end[0][tf.newaxis]
    #end = start_end[1][tf.newaxis]
    start = tf.constant([self.start_voc_id],dtype=tf.int64)
    end = tf.constant([self.end_voc_id],dtype=tf.int64)

    # `tf.TensorArray` is required here (instead of a python list) so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)

    for i in tf.range(max_length):
      output = tf.transpose(output_array.stack())
      predictions, _ = self.transformer([encoder_input, output], training=False)

      # select the last token from the seq_len dimension
      predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

      predicted_id = tf.argmax(predictions, axis=-1)

      # concatentate the predicted_id to the output which is given to the decoder
      # as its input.
      output_array = output_array.write(i+1, predicted_id[0])

      if predicted_id == end:
        break

    output = tf.transpose(output_array.stack())
    # output.shape (1, tokens)
    #text = tokenizers.en.detokenize(output)[0]  # shape: ()

    #tokens = tokenizers.en.lookup(output)[0]

    # `tf.function` prevents us from using the attention_weights that were
    # calculated on the last iteration of the loop. So recalculate them outside
    # the loop.
    _, attention_weights = self.transformer([encoder_input, output[:,:-1]], training=False)

    #return text, tokens, attention_weights
    return output,attention_weights
  
translator = Translator(
  transformer,
  max_out_length=max_length_targ,
  start_voc_id=targ_lang.word_index['<start>'],
  end_voc_id=targ_lang.word_index['<end>'],
)

def print_translation(sentence, tokens, ground_truth):
  print(f'{"Input:":15s}: {sentence}')
  print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
  print(f'{"Ground truth":15s}: {ground_truth}')


#sentence = "este é um problema que temos que resolver."
#ground_truth = "this is a problem we have to solve ."
#
#translated_text, translated_tokens, attention_weights = translator(
#    tf.constant(sentence))
#print_translation(sentence, translated_text, ground_truth)
#
#sentence = "os meus vizinhos ouviram sobre esta ideia."
#ground_truth = "and my neighboring homes heard about this idea ."
#
#translated_text, translated_tokens, attention_weights = translator(
#    tf.constant(sentence))
#print_translation(sentence, translated_text, ground_truth)

#sentence = "vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram."
#ground_truth = "so i \'ll just share with you some stories very quickly of some magical things that have happened ."
#
#translated_text, translated_tokens, attention_weights = translator(
#    tf.constant(sentence))
#print_translation(sentence, translated_text, ground_truth)

def plot_attention_head(in_tokens, translated_tokens, attention):
  # The plot is of the attention when a token was generated.
  # The model didn't generate `<START>` in the output. Skip it.
  translated_tokens = translated_tokens[1:]

  ax = plt.gca()
  ax.matshow(attention)
  ax.set_xticks(range(len(in_tokens)))
  ax.set_yticks(range(len(translated_tokens)))

  #labels = [label.decode('utf-8') for label in in_tokens.numpy()]
  labels = in_tokens
  ax.set_xticklabels(
      labels, rotation=90)

  #labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
  labels = translated_tokens
  ax.set_yticklabels(labels)

head = 0
# shape: (batch=1, num_heads, seq_len_q, seq_len_k)
#attention_heads = tf.squeeze(
#  attention_weights['decoder_layer4_block2'], 0)
#attention = attention_heads[head]
#attention.shape

#in_tokens = tf.convert_to_tensor([sentence])
#in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
#in_tokens = tokenizers.pt.lookup(in_tokens)[0]
#in_tokens

#plot_attention_head(in_tokens, translated_tokens, attention)

def plot_attention_weights(sentence, translated_tokens, attention_heads):
  #in_tokens = tf.convert_to_tensor([sentence])
  #in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
  #in_tokens = tokenizers.pt.lookup(in_tokens)[0]
  in_tokens = sentence

  fig = plt.figure(figsize=(16, 8))

  for h, head in enumerate(attention_heads):
    ax = fig.add_subplot(2, 4, h+1)

    plot_attention_head(in_tokens, translated_tokens, head)

    ax.set_xlabel(f'Head {h+1}')

  plt.tight_layout()
  plt.show()

#plot_attention_weights(sentence, translated_tokens,
#                       attention_weights['decoder_layer4_block2'][0])

#sentence = "Eu li sobre triceratops na enciclopédia."
#ground_truth = "I read about triceratops in the encyclopedia."

#translated_text, translated_tokens, attention_weights = translator(
#    tf.constant(sentence))
#print_translation(sentence, translated_text, ground_truth)
#predict, attention_weights = translator(
#    tf.constant(sentence))
#plot_attention_weights(sentence, translated_tokens,
#                       attention_weights['decoder_layer4_block2'][0])

for i in range(10):
    idx = np.random.randint(0,len(input_tensor))
    question = input_tensor[idx]
    predict,attention_plot = translator(tf.constant(question))
    predict = predict[0].numpy()
    answer = target_tensor[idx]
    sentence = inp_lang.sequences_to_texts([question])[0]
    predicted_sentence = targ_lang.sequences_to_texts([predict])[0]
    target_sentence = targ_lang.sequences_to_texts([answer])[0]
    print('Input:',sentence)
    print('Predict:',predicted_sentence)
    print('Target:',target_sentence)
    print(' ')
    
    sentence = sentence.split(' ')
    predicted_sentence = predicted_sentence.split(' ')
    #print(attention_plot)
    #attention_plot = tf.squeeze(attention_plot,0)
    #attention_plot = attention_plot[:,:len(predicted_sentence), :len(sentence)]
    
    #plot_attention_weights(sentence, predicted_sentence, attention_plot['decoder_layer4_block2'][0])


#class ExportTranslator(tf.Module):
#  def __init__(self, translator):
#    self.translator = translator
#
#  @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
#  def __call__(self, sentence):
#    (result, 
#     tokens,
#     attention_weights) = self.translator(sentence, max_length=100)
#
#    return result
  
#translator = ExportTranslator(translator)

#translator("este é o primeiro livro que eu fiz.").numpy()

#tf.saved_model.save(translator, export_dir='translator')

#reloaded = tf.saved_model.load('translator')

#reloaded("este é o primeiro livro que eu fiz.").numpy()
