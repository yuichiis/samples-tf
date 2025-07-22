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


def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]    # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth  # (1, depth/2)

  angle_rates = 1 / (10000**depths)               # (1, depth/2)
  angle_rads = positions * angle_rates            # (length, depth/2)

  pos_encoding = np.concatenate(                  # (length, depth/2*2)
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)  # (length, depth)


class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model, kernel_initializer=None):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(
        vocab_size, d_model,
        mask_zero=True,
        embeddings_initializer=kernel_initializer) 
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x


class BaseAttention(tf.keras.layers.Layer):
  def __init__(self,
      kernel_initializer=None,
      bias_initializer=None,
      **kwargs
  ):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
       **kwargs
    )
    self.layernorm = tf.keras.layers.LayerNormalization(
       gamma_initializer=kernel_initializer,
       beta_initializer=bias_initializer,
    )
    self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class FeedForward(tf.keras.layers.Layer):
  def __init__(self,
      d_model, dff,
      #dropout_rate=0.1,
      kernel_initializer=None,
      bias_initializer=None,
  ):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(
          dff,
          activation='relu',
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer,
      ),
      tf.keras.layers.Dense(
          d_model,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer,
      ),
      #tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization(
      gamma_initializer=kernel_initializer,
      beta_initializer=bias_initializer,
    )

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x


# FC = Fully connected (dense) layer
# EO = Encoder output
# H = hidden state
# X = input to the decoder
# And the pseudo-code:

# score = FC(tanh(FC(EO) + FC(H)))
# attention weights = softmax(score, axis = 1). Softmax by default is applied on the last axis but here we want to apply it on the 1st axis, since the shape of score is (batch_size, max_length, hidden_size). Max_length is the length of our input. Since we are trying to assign a weight to each input, softmax should be applied on that axis.
# context vector = sum(attention weights * EO, axis = 1). Same reason as above for choosing axis as 1.
# embedding output = The input to the decoder X is passed through an embedding layer.
# merged vector = concat(embedding output, context vector)
# This merged vector is then given to the GRU
# The shapes of all the vectors at each step have been specified in the comments in the code:

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, 
      d_model, num_heads, dff,
      #dropout_rate=0.1,
      kernel_initializer=None,
      bias_initializer=None,
  ):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        #dropout=dropout_rate,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )

    self.ffn = FeedForward(
        d_model, dff,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x

class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, 
      num_layers, d_model, num_heads,
      dff, vocab_size,
      #dropout_rate=0.1,
      kernel_initializer=None,
      bias_initializer=None,
    ):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        kernel_initializer=kernel_initializer,
    )

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     #dropout_rate=dropout_rate,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer,
        )
        for _ in range(num_layers)]
    #self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

    # Add dropout.
    #x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.




class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
      *,
      d_model,
      num_heads,
      dff,
      #dropout_rate=0.1,
      kernel_initializer=None,
      bias_initializer=None,
  ):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        #dropout=dropout_rate,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        #dropout=dropout_rate,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )

    self.ffn = FeedForward(d_model, dff,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x

class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, 
      num_layers, d_model, num_heads, dff, vocab_size,
      #dropout_rate=0.1,
      kernel_initializer=None,
      bias_initializer=None,
  ):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        kernel_initializer=kernel_initializer,
    )
    #self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            #dropout_rate=dropout_rate,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context):
    # `x` is token-IDs shape (batch, target_seq_len)
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

    #x = self.dropout(x)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context)

    self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    # The shape of x is (batch_size, target_seq_len, d_model).
    return x


class Transformer(tf.keras.Model):
  def __init__(self, *, 
        num_layers, d_model, num_heads, dff,
        input_vocab_size, target_vocab_size,
        #dropout_rate=0.1,
        kernel_initializer=None,
        bias_initializer=None,
  ):
    super().__init__()
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           #dropout_rate=dropout_rate,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
    )

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           #dropout_rate=dropout_rate,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
    )

    self.final_layer = tf.keras.layers.Dense(
        target_vocab_size,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )

  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    context, x  = inputs

    context = self.encoder(context)  # (batch_size, context_len, d_model)

    x = self.decoder(x, context)  # (batch_size, target_len, d_model)

    # Final linear layer output.
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits

class Translator(tf.Module):
  def __init__(self, transformer,
      max_out_length=None, start_voc_id=None, end_voc_id=None):
    #self.tokenizers = tokenizers
    self.transformer = transformer
    self.max_out_length = max_out_length
    self.start_voc_id = start_voc_id
    self.end_voc_id = end_voc_id

  def __call__(self, sentence):
    # The input sentence is Portuguese, hence adding the `[START]` and `[END]` tokens.
    assert isinstance(sentence, tf.Tensor)
    #if len(sentence.shape) == 0:
    #  sentence = sentence[tf.newaxis]
    #
    #sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

    encoder_input = tf.expand_dims(sentence,axis=0)

    # As the output language is English, initialize the output with the
    # English `[START]` token.
    #start_end = self.tokenizers.en.tokenize([''])[0]
    #start = start_end[0][tf.newaxis]
    #end = start_end[1][tf.newaxis]
    start = tf.constant([self.start_voc_id],dtype=tf.int64)
    end = tf.constant([self.end_voc_id],dtype=tf.int64)

    # `tf.TensorArray` is required here (instead of a Python list), so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)

    for i in tf.range(self.max_out_length):
      output = tf.transpose(output_array.stack())
      predictions = self.transformer([encoder_input, output], training=False)

      # Select the last token from the `seq_len` dimension.
      predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

      predicted_id = tf.argmax(predictions, axis=-1)

      # Concatenate the `predicted_id` to the output which is given to the
      # decoder as its input.
      output_array = output_array.write(i+1, predicted_id[0])

      if predicted_id == end:
        break

    output = tf.transpose(output_array.stack())
    # The output shape is `(1, tokens)`.
    #text = tokenizers.en.detokenize(output)[0]  # Shape: `()`.

    #tokens = tokenizers.en.lookup(output)[0]

    # `tf.function` prevents us from using the attention_weights that were
    # calculated on the last iteration of the loop.
    # So, recalculate them outside the loop.
    self.transformer([encoder_input, output[:,:-1]], training=False)
    attention_weights = self.transformer.decoder.last_attn_scores

    #return text, tokens, attention_weights
    return output,attention_weights

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def plot_attention_head(in_tokens, translated_tokens, attention):
  # The model didn't generate `<START>` in the output. Skip it.
  translated_tokens = translated_tokens[1:]

  ax = plt.gca()
  ax.matshow(attention)
  ax.set_xticks(range(len(in_tokens)))
  ax.set_yticks(range(len(translated_tokens)))

  #labels = [label.decode('utf-8') for label in in_tokens]
  labels = in_tokens
  ax.set_xticklabels(
      labels, rotation=90)

  #labels = [label.decode('utf-8') for label in translated_tokens]
  labels = translated_tokens
  ax.set_yticklabels(labels)

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


def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss


def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)



#################################################################
#  Parameters
#################################################################

num_layers = 1#4#6
d_model = 4#128#512 # embedding_dim
dff = 4#512 # units
num_heads = 2#8
dropout_rate = 0.1#0.1

num_examples = 20000 #30000 #5000 #10000
num_words = 1024 #None #128
EPOCHS = 10
BATCH_SIZE = 64
#embedding_dim = 256
#units = 1024



from functools import reduce
from operator import mul
vocab_size = 16
d_model = 4
context_len = 3
input_vocab_size = vocab_size
target_vocab_size = vocab_size
shape = [1,context_len]
v_shape = [1,context_len, d_model]
inp = tf.ones(shape, dtype=tf.int32)
inp = tf.reshape(tf.range(1, 1+reduce(mul, shape), dtype=tf.int32),shape)
targ = tf.reshape(tf.range(1, 1+reduce(mul, shape), dtype=tf.int32),shape)
inp_seq = tf.Variable(inp)  # (batch_size, context_len, d_model)
targ_seq = tf.Variable(targ)  # (batch_size, context_len, d_model)
inp_vec = tf.reshape(tf.range(1, 1+reduce(mul, v_shape), dtype=tf.float32),v_shape)

#posEmb = PositionalEmbedding(vocab_size, d_model, kernel_initializer='ones')
#x = posEmb(inp_seq)
#print(x)

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    #dropout_rate=dropout_rate,
    kernel_initializer='ones',
    bias_initializer='zeros',
)
#predict = transformer([inp_seq, targ_seq], training=True)
#with tf.GradientTape() as tape:
#  predict = transformer([inp_seq, targ_seq], training=True)
#print(predict)
#params = transformer.trainable_variables
#grads = tape.gradient(predict,params)
#print(len(grads))
#for i,(param,grad) in enumerate(zip(params,grads)):
#   print(i,':',param.name[-16:],':',grad)

input_tensor_train = np.array([
    [10,2,3,4,11,0,0,0,],
    [10,4,3,2,1,11,0,0,],
])
target_tensor_train = np.array([
    [10,1,2,1,2,3,11,0,],
    [10,3,4,3,4,1,11,0,],
])
label_tensor_train = np.array([
    [1,2,1,2,3,11,0,0,],
    [3,4,3,4,1,11,0,0,],
])

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])
history = transformer.fit((input_tensor_train,target_tensor_train),label_tensor_train,
                epochs=100,batch_size=BATCH_SIZE,verbose=0)
test_inputs = np.array([
    [10,2,3,4,11,0,0,0,],
])
test_targets = np.array([
    [10,1,2,1,2,3,11,0,],
])
preds = transformer.predict([test_inputs, test_targets])
print("Inputs:",test_inputs.shape)
print("targets:",test_targets.shape)
print("Preds :",preds.shape)
print(np.argmax(preds, axis=-1))


exit()
#encoder = Encoder(num_layers=num_layers, d_model=d_model,
#        num_heads=num_heads, dff=dff,
#        vocab_size=input_vocab_size,
#        #dropout_rate=dropout_rate,
#        kernel_initializer='ones',
#        bias_initializer='zeros',
#)
#predict = encoder(inp_seq, training=True)
#print(predict)
#decoder = Decoder(num_layers=num_layers, d_model=d_model,
#        num_heads=num_heads, dff=dff,
#        vocab_size=target_vocab_size,
#        #dropout_rate=dropout_rate,
#        kernel_initializer='ones',
#        bias_initializer='zeros',
#)
#predict = decoder(inp_seq, inp_vec, training=True)
#print(predict)
#encoder_layer = EncoderLayer(d_model=d_model,
#    num_heads=num_heads,
#    dff=dff,
#    dropout_rate=dropout_rate,
#    kernel_initializer='ones',
#    bias_initializer='zeros',
#)
#predict = encoder_layer(inp_vec, training=True)
#print(predict)

exit()

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
print("train on:",len(input_tensor_train))
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

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)


transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    dropout_rate=dropout_rate)

print("Compile model...")
transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])

print("Train model...")
history = transformer.fit((input_tensor_train,target_tensor_train),label_tensor_train,
                epochs=EPOCHS,batch_size=BATCH_SIZE,
                validation_data=((input_tensor_val,target_tensor_val),label_tensor_val))

print('trainable_weights=',len(transformer.trainable_weights))
#for weight in transformer.trainable_weights:
#   print(weight.name)
#exit()

plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['masked_accuracy'],label='accuracy')
plt.plot(history.history['val_loss'],label='val_loss')
plt.plot(history.history['val_masked_accuracy'],label='val_accuracy')
plt.legend()
plt.show()

translator = Translator(
    transformer,
    max_out_length=max_length_targ,
    start_voc_id=targ_lang.word_index['<start>'],
    end_voc_id=targ_lang.word_index['<end>'],
)

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
    attention_plot = tf.squeeze(attention_plot,0)
    attention_plot = attention_plot[:,:len(predicted_sentence), :len(sentence)]
    plot_attention_weights(sentence, predicted_sentence, attention_plot)

