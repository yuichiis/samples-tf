# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
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

    def tokenize(self, lang):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')
        lang_tokenizer.fit_on_texts(lang)
        tensor = lang_tokenizer.texts_to_sequences(lang)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')
        return tensor, lang_tokenizer

    def load_dataset(self, path, num_examples=None):
        # creating cleaned input, output pairs
        targ_lang, inp_lang = self.create_dataset(path, num_examples)

        input_tensor, inp_lang_tokenizer = self.tokenize(inp_lang)
        target_tensor, targ_lang_tokenizer = self.tokenize(targ_lang)

        return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

    def convert(self, lang, tensor):
        for t in tensor:
            if t!=0:
                print ("%d ----> %s" % (t, lang.index_word[t]))


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

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units):
    super(Encoder, self).__init__()
    #self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self,batch_sz):
    return tf.zeros((batch_sz, self.enc_units))




class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights



class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units):
    super(Decoder, self).__init__()
    #self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights


#class Seq2seq(tf.keras.Model):
#    def __init__(self,
#        vocab_inp_size, vocab_tar_size, embedding_dim, units,
#        **kwargs
#    ):
#        super(Seq2seq, self).__init__(**kwargs)
#        self.encoder = Encoder(vocab_inp_size, embedding_dim, units)
#        self.decoder = Decoder(vocab_tar_size, embedding_dim, units)
#        self.attention_layer = BahdanauAttention(10)
#
#    def call(
#        self,
#        inputs,
#        training=None,
#        mask=None,
#        ):
#        inp, targ = inputs
#        batch_size = tf.shape(inp)[0]
#        enc_hidden = self.encoder.initialize_hidden_state(batch_size)
#        enc_output, enc_hidden = self.encoder(inp, enc_hidden)
#        dec_hidden = enc_hidden
#        dec_input = tf.expand_dims(
#            tf.repeat([targ_lang.word_index['<start>']], repeats=[batch_size]), 1)
#        outs = []
#        # Teacher forcing - feeding the target as the next input
#        for t in range(targ.shape[1]):
#            # passing enc_output to the decoder
#            predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
#            outs.append(predictions)
#            #loss += loss_function(targ[:, t], predictions)
#
#            # using teacher forcing
#            dec_input = tf.expand_dims(targ[:, t], 1)
#
#        outputs = tf.stack(outs,axis=1)
#        return outputs
#
#    def train_step(
#        self,
#        train_data: tuple
#    ):
#        '''train step callback'''
#        inputs,trues = train_data
#
#        with tf.GradientTape() as tape:
#            outputs = self(train_data,training=True)
#            loss = self.compiled_loss(
#                trues,outputs,
#                regularization_losses=self.losses)
#
#        variables = self.trainable_variables
#
#        gradients = tape.gradient(loss, variables)
#
#        self.optimizer.apply_gradients(zip(gradients, variables))
#
#        self.compiled_metrics.update_state(trues, outputs)
#        # Return a dict mapping metric names to current value
#        return {m.name: m.result() for m in self.metrics}
#
#    def test_step(self, data):
#        """
#        The logic for one evaluation step.
#        """
#        x, y = data
#
#        y_pred = self(data, training=False)
#        # Updates stateful loss metrics.
#        self.compiled_loss(
#            y, y_pred, regularization_losses=self.losses)
#
#        self.compiled_metrics.update_state(y, y_pred)
#        return {m.name: m.result() for m in self.metrics}
#
#    def translate(self, sentence, dataset):
#        result, sentence, attention_plot = self.evaluate_sentence(sentence, dataset)
#
#        print('Input: %s' % (sentence))
#        print('Predicted translation: {}'.format(result))
#
#        attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
#        self.plot_attention(attention_plot, sentence.split(' '), result.split(' '))
#
#    def evaluate_sentence(self,sentence, dataset):
#        attention_plot = np.zeros((max_length_targ, max_length_inp))
#
#        sentence = dataset.preprocess_sentence(sentence)
#
#        inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
#        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
#                                                         maxlen=max_length_inp,
#                                                         padding='post')
#        inputs = tf.convert_to_tensor(inputs)
#
#        result = ''
#
#        hidden = [tf.zeros((1, units))]
#        enc_out, enc_hidden = self.encoder(inputs, hidden)
#
#        dec_hidden = enc_hidden
#        dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
#
#        for t in range(max_length_targ):
#            predictions, dec_hidden, attention_weights = self.decoder(dec_input,
#                                                         dec_hidden,
#                                                         enc_out)
#
#        # storing the attention weights to plot later on
#        attention_weights = tf.reshape(attention_weights, (-1, ))
#        attention_plot[t] = attention_weights.numpy()
#
#        predicted_id = tf.argmax(predictions[0]).numpy()
#
#        result += targ_lang.index_word[predicted_id] + ' '
#
#        if targ_lang.index_word[predicted_id] == '<end>':
#            return result, sentence, attention_plot
#
#        # the predicted ID is fed back into the model
#        dec_input = tf.expand_dims([predicted_id], 0)
#
#        return result, sentence, attention_plot
#
#    # function for plotting the attention weights
#    def plot_attention(attention, sentence, predicted_sentence):
#        fig = plt.figure(figsize=(10,10))
#        ax = fig.add_subplot(1, 1, 1)
#        ax.matshow(attention, cmap='viridis')
#
#        fontdict = {'fontsize': 14}
#
#        ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
#        ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)
#
#        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#
#        plt.show()
#

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


def create_seq2seq(
        vocab_inp_size, vocab_tar_size,
        max_length_inp, max_length_targ,
        embedding_dim, units,
        targ_lang, batch_size,
    ):
    encoder = Encoder(vocab_inp_size, embedding_dim, units)
    decoder = Decoder(vocab_tar_size, embedding_dim, units)
    attention_layer = BahdanauAttention(10)

    inp = tf.keras.Input(shape=(max_length_inp,))
    targ = tf.keras.Input(shape=(max_length_targ,))

    batch_size = tf.shape(inp)[0]

    enc_hidden = encoder.initialize_hidden_state(batch_size)
    enc_output, enc_hidden = encoder(inp, enc_hidden)
    dec_hidden = enc_hidden
    r = tf.repeat([targ_lang.word_index['<start>']], repeats=[batch_size])
    print('r',r)
    dec_input = tf.expand_dims(r , 1)
    #dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * batch_size, 1)
    print('dec_input',dec_input)
    outs = []
    # Teacher forcing - feeding the target as the next input
    for t in range(targ.shape[1]):
        # passing enc_output to the decoder
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
        outs.append(predictions)
        #loss += loss_function(targ[:, t], predictions)
        # using teacher forcing
        dec_input = tf.expand_dims(targ[:, t], 1)

    outputs = tf.stack(outs,axis=1)
    model = tf.keras.Model([inp,targ],outputs)
    return model



#en_sentence = u"May I borrow this book?"
#sp_sentence = u"¿Puedo tomar prestado este libro?"
#print(preprocess_sentence(en_sentence))
#print(preprocess_sentence(sp_sentence).encode('utf-8'))

#en, sp = create_dataset(path_to_file, None)
#print(en[-1])
#print(sp[-1])
#print('en=',len(en))
#print('sp=',len(sp))

# Try experimenting with the size of that dataset
dataset = EngFraDataset()
path_to_file = dataset.download()
num_examples = 5000 #30000
print("Generating data...")
input_tensor, target_tensor, inp_lang, targ_lang = dataset.load_dataset(path_to_file, num_examples)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

# Creating training and validation sets using an 80-20 split
#input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)
split_at = len(input_tensor) - len(input_tensor) // 10 * 2
input_tensor_train,  input_tensor_val  = (input_tensor[:split_at], input_tensor[split_at:])
target_tensor_train, target_tensor_val = (target_tensor[:split_at],target_tensor[split_at:])

# Show length
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

#print ("Input Language; index to word mapping")
#convert(inp_lang, input_tensor_train[0])
#print ()
#print ("Target Language; index to word mapping")
#convert(targ_lang, target_tensor_train[0])

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 128#256
units = 512#1024
print("Input  length: ",max_length_inp)
print("Target length: ",max_length_targ)
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
print("Input vocabulary size : ",vocab_inp_size)
print("Target vocabulary size: ",vocab_tar_size)

#dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
#dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

#example_input_batch, example_target_batch = next(iter(dataset))
#example_input_batch.shape, example_target_batch.shape



#encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# sample input
#sample_hidden = encoder.initialize_hidden_state()
#sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
#print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
#print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))



#attention_layer = BahdanauAttention(10)
#attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

#print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
#print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

#decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

#sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
#                                      sample_hidden, sample_output)

#print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))



#optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

checkpoint_dir = './training_checkpoints'
#checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
#checkpoint = tf.train.Checkpoint(optimizer=optimizer,
#                                 encoder=encoder,
#                                 decoder=decoder)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

EPOCHS = 10
#seq2seq = Seq2seq(
#    vocab_inp_size, vocab_tar_size, embedding_dim, units,
#)
seq2seq = create_seq2seq(
        vocab_inp_size, vocab_tar_size,
        max_length_inp, max_length_targ,
        embedding_dim, units,
        targ_lang, BATCH_SIZE,
)
seq2seq.summary()

print("Compile model...")
seq2seq.compile(
    #loss='sparse_categorical_crossentropy',
    #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    loss=loss_function,
    optimizer='adam',
    metrics=['accuracy'],
    )

print("Train model...")
history = seq2seq.fit(
    [input_tensor_train,target_tensor_train],
    target_tensor_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(input_tensor_val,target_tensor_val),
)

seq2seq.load_weights(checkpoint_dir)
# restoring the latest checkpoint in checkpoint_dir
#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
#print(sp[num_examples-1])
#print(en[num_examples-1])
#print(sp[num_examples-10])
#print(en[num_examples-10])
#print(sp[num_examples-100])
#print(en[num_examples-100])
seq2seq.translate(u'el cogio el telefono.')
print('correct: he hung up.')
seq2seq.translate(u'confien.')
print('correct: have faith.')
seq2seq.translate(u'llega a tiempo.')
print('correct: be on time.')

#translate(u'hace mucho frio aqui.')
#translate(u'esta es mi vida.')
#translate(u'¿todavia estan en casa?')
# wrong translation
#translate(u'trata de averiguarlo.')
