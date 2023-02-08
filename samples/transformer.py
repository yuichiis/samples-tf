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
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
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

    #def seq2str(self,sequence,lang):
    #    result = ''
    #    for word_id in sequence:
    #        if word_id == 0:
    #            break
    #        else:
    #            word = lang.index_word[word_id]
    #            if result=='':
    #                result = word
    #            else:
    #                result = result+' '+word
    #            if word == '<end>':
    #                return result
    #    return result

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

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)

#pos_encoding = positional_encoding(length=2048, depth=512)

# Check the shape.
#print(pos_encoding.shape)

# Plot the dimensions.
#plt.pcolormesh(pos_encoding.numpy().T, cmap='RdBu')
#plt.ylabel('Depth')
#plt.xlabel('Position')
#plt.colorbar()
#plt.show()


#pos_encoding/=tf.norm(pos_encoding, axis=1, keepdims=True)
#p = pos_encoding[1000]
#dots = tf.einsum('pd,d -> p', pos_encoding, p)
#plt.subplot(2,1,1)
#plt.plot(dots)
#plt.ylim([0,1])
#plt.plot([950, 950, float('nan'), 1050, 1050],
#         [0,1,float('nan'),0,1], color='k', label='Zoom')
#plt.legend()
#plt.subplot(2,1,2)
#plt.plot(dots)
#plt.xlim([950, 1050])
#plt.ylim([0,1])
#plt.show()


class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
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
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
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
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

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
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x

class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(
        vocab_size=vocab_size, d_model=d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.




class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x

class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
               dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                             d_model=d_model)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context):
    # `x` is token-IDs shape (batch, target_seq_len)
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

    x = self.dropout(x)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context)

    self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    # The shape of x is (batch_size, target_seq_len, d_model).
    return x


class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           dropout_rate=dropout_rate)

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

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


#def train_step(inp, targ, enc_hidden):
#  loss = 0
#
#  with tf.GradientTape() as tape:
#    enc_output, enc_hidden = encoder(inp, enc_hidden)
#
#    dec_hidden = enc_hidden
#
#    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)
#
#    # Teacher forcing - feeding the target as the next input
#    for t in range(1, targ.shape[1]):
#      # passing enc_output to the decoder
#      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
#
#      loss += loss_function(targ[:, t], predictions)
#
#      # using teacher forcing
#      dec_input = tf.expand_dims(targ[:, t], 1)
#
#  batch_loss = (loss / int(targ.shape[1]))
#
#  variables = encoder.trainable_variables + decoder.trainable_variables
#
#  gradients = tape.gradient(loss, variables)
#
#  optimizer.apply_gradients(zip(gradients, variables))
#
#  return batch_loss

#EPOCHS = 10

#for epoch in range(EPOCHS):
#  start = time.time()
#
#  enc_hidden = encoder.initialize_hidden_state()
#  total_loss = 0
#
#  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
#    batch_loss = train_step(inp, targ, enc_hidden)
#    total_loss += batch_loss
#
#    if batch % 100 == 0:
#      print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
#                                                   batch,
#                                                   batch_loss.numpy()))
#  # saving (checkpoint) the model every 2 epochs
#  if (epoch + 1) % 2 == 0:
#    checkpoint.save(file_prefix = checkpoint_prefix)
#
#  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
#                                      total_loss / steps_per_epoch))
#  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

#en_sentence = u"May I borrow this book?"
#sp_sentence = u"¿Puedo tomar prestado este libro?"
#print(preprocess_sentence(en_sentence))
#print(preprocess_sentence(sp_sentence).encode('utf-8'))

#en, sp = create_dataset(path_to_file, None)
#print(en[-1])
#print(sp[-1])
#print('en=',len(en))
#print('sp=',len(sp))


num_layers = 4
d_model = 128 # embedding_dim
dff = 512
num_heads = 8
dropout_rate = 0.1

num_examples = 5000 #10000 #30000
num_words = None #128
EPOCHS = 10
#BATCH_SIZE = 64
#embedding_dim = 256
#units = 1024






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

label_tensor_train  = input_tensor[:split_at]
lebel_len,lebel_words = label_tensor_train.shape
label_tensor_train  = label_tensor_train[:,:lebel_words-1]
filler = np.zeros(lebel_len,dtype=label_tensor_train.dtype).reshape(lebel_len,1)
label_tensor_train  = np.append(label_tensor_train,filler,axis=1)


# Show length
print('input=',input_tensor_train.shape)
print('target=',target_tensor_train.shape)
print('label=',label_tensor_train.shape)
print('val_input=',input_tensor_val.shape)
print('val_target=', target_tensor_val.shape)



##### Test PositionalEmbedding
#embed_pt = PositionalEmbedding(vocab_size=input_vocab_size, d_model=512)
#embed_en = PositionalEmbedding(vocab_size=target_vocab_size, d_model=512)
##
#pt_emb = embed_pt(input_tensor_train)
#en_emb = embed_en(target_tensor_train)
##
#print(en_emb._keras_mask)
#
#exit()


#print ("Input Language; index to word mapping")
#convert(inp_lang, input_tensor_train[0])
#print ()
#print ("Target Language; index to word mapping")
#convert(targ_lang, target_tensor_train[0])

#BUFFER_SIZE = len(input_tensor_train)
#steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
#print("num_examples:",num_examples)
#print('num_words:',num_words)
#print("epoch:",EPOCHS)
#print("batch_size:",BATCH_SIZE)
#print("embedding_dim:",embedding_dim)
#print("units: ",units)
#print("Input  length:",max_length_inp)
#print("Target length:",max_length_targ)
#print("Input  word dictionary: %d(%d)" % (input_vocab_size,len(inp_lang.index_word)+1))
#print("Target word dictionary: %d(%d)" % (target_vocab_size,len(targ_lang.index_word)+1))

#example_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
#example_dataset = example_dataset.batch(BATCH_SIZE, drop_remainder=True)
#
#example_input_batch, example_target_batch = next(iter(example_dataset))
#example_input_batch.shape, example_target_batch.shape

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

#seq2seq = Seq2seq(
#    input_length=max_length_inp,
#    input_vocab_size=input_vocab_size,
#    output_length=max_length_targ,
#    target_vocab_size=target_vocab_size,
#    embedding_dim=embedding_dim,
#    units=units,
#    start_voc_id=targ_lang.word_index['<start>'],
#    end_voc_id=targ_lang.word_index['<end>'],
#)


print("Compile model...")
transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])

#seq2seq.compile(
#    #loss='sparse_categorical_crossentropy',
#    #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#    loss=loss_function,
#    optimizer='adam',
#    metrics=['accuracy'],
#    )
#
##exit()

transformer.fit((input_tensor_train,target_tensor_train),label_tensor_train,
                epochs=EPOCHS,
                validation_data=(input_tensor_val,target_tensor_val))

#
#print("Train model...")
#history = seq2seq.fit(
#    input_tensor_train,
#    target_tensor_train,
#    batch_size=BATCH_SIZE,
#    epochs=EPOCHS,
#    validation_data=(input_tensor_val,target_tensor_val),
#    #callbacks=[model_checkpoint_callback],
#)

#seq2seq.load_weights(checkpoint_dir)

# restoring the latest checkpoint in checkpoint_dir
#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
#print(sp[num_examples-1])
#print(en[num_examples-1])
#print(sp[num_examples-10])
#print(en[num_examples-10])
#print(sp[num_examples-100])
#print(en[num_examples-100])
#seq2seq.translate(u'el cogio el telefono.',dataset)
#print('correct: he hung up.')
#seq2seq.translate(u'confien.',dataset)
#print('correct: have faith.')
#seq2seq.translate(u'llega a tiempo.',dataset)
#print('correct: be on time.')

#translate(u'hace mucho frio aqui.')
#translate(u'esta es mi vida.')
#translate(u'¿todavia estan en casa?')
# wrong translation
#translate(u'trata de averiguarlo.')
#for i in range(10):
#    idx = np.random.randint(0,len(input_tensor))
#    question = input_tensor[idx]
#    #input = question.reshape(1,max_length_inp)
#    #input = keras.utils.to_categorical(
#    #    input.reshape(input.size,),
#    #    num_classes=len(input_voc)
#    #    ).reshape(input.shape[0],input.shape[1],len(input_voc))
#
#    #predict = model.predict(input)
#    #predict_seq = np.argmax(predict[0].reshape(output_length,len(target_dic)),axis=1)
#    sentence = dataset.seq2str(question,inp_lang)
#    predict_seq = seq2seq.translate(sentence, dataset);
#    answer = target_tensor[idx]
#    sentence = dataset.seq2str(answer,targ_lang)
#    print('Target: %s' % (sentence))
#    print()
#for i in range(10):
#    idx = np.random.randint(0,len(input_tensor))
#    question = input_tensor[idx]
#    predict, attention_plot = seq2seq.evaluate_sequence([question])
#    answer = target_tensor[idx]
#    sentence = inp_lang.sequences_to_texts([question])[0]
#    predicted_sentence = targ_lang.sequences_to_texts([predict])[0]
#    target_sentence = targ_lang.sequences_to_texts([answer])[0]
#    print('Input:',sentence)
#    print('Predict:',predicted_sentence)
#    print('Target:',target_sentence)
#    print()
#    #attention_plot = attention_plot[:len(predicted_sentence.split(' ')), :len(sentence.split(' '))]
#    seq2seq.plot_attention(attention_plot, sentence.split(' '), predicted_sentence.split(' '))

#plt.plot(history.history['loss'],label='loss')
#plt.plot(history.history['accuracy'],label='accuracy')
#plt.plot(history.history['val_loss'],label='val_loss')
#plt.plot(history.history['val_accuracy'],label='val_accuracy')
#plt.legend()
#plt.show()
