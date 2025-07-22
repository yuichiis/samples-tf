# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pickle
#from packaging import version
from distutils.version import LooseVersion, StrictVersion
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

print('tensorflow: %s, keras: %s' % (tf.__version__, keras.__version__))
if StrictVersion(tf.__version__) < StrictVersion("2.2.0"):
    raise Exception('tensorflow v2.2.0 or later is required.')

TRAINING_SIZE = 5000
DIGITS = 3


class Encoder(keras.Model):
    '''
    encoder
    '''
    def __init__(
        self,
        input_length: int,
        vocab_size: int,
        word_vect_size: int,
        recurrent_units: int,
        **kwargs
    ):
        '''encoder'''
        super(Encoder, self).__init__(**kwargs)
        #self.input_shape=[input_length]
        self.vocab_size = vocab_size
        self.word_vect_size = word_vect_size
        self.recurrent_units = recurrent_units

        self.embedding = keras.layers.Embedding(vocab_size, word_vect_size)
        self.rnn = keras.layers.GRU(recurrent_units,
            return_state=True,return_sequences=True,)
        #self.rnn = keras.layers.LSTM(recurrent_units,
        #    return_state=True,return_sequences=True,)
        #self.rnn = keras.layers.SimpleRNN(recurrent_units,
        #    return_state=True,return_sequences=True,)

    def call(
        self,
        inputs: ndarray,
        training: bool,
        initial_state: tuple=None,
        **kwargs) -> tuple:
        '''forward'''
        wordvect = self.embedding(inputs)
        states = self.rnn(wordvect,training=training,initial_state=initial_state)
        outputs = states.pop(0)
        return (outputs, states)


class Decoder(keras.Model):
    '''
    Decoder
    '''
    def __init__(
        self,
        input_length: int,
        vocab_size: int,
        word_vect_size: int,
        recurrent_units: int,
        **kwargs
        ):
        '''
        Decoder
        '''
        super(Decoder, self).__init__(**kwargs)
        #self.input_shape=[input_length]
        self.vocab_size = vocab_size
        self.word_vect_size = word_vect_size
        self.recurrent_size = recurrent_units

        self.embedding = keras.layers.Embedding(vocab_size, word_vect_size)
        self.rnn = keras.layers.GRU(recurrent_units,
            return_state=True,
            return_sequences=True,)
        #self.rnn = keras.layers.LSTM(recurrent_units,
        #    return_state=True,
        #    return_sequences=True,)
        #self.rnn = keras.layers.SimpleRNN(recurrent_units,
        #    return_state=True,
        #    return_sequences=True,)
        self.attention = keras.layers.Attention()
        self.concat = keras.layers.Concatenate()
        self.dense = keras.layers.Dense(vocab_size)

    def call(
        self,
        inputs: ndarray,
        training: bool,
        initial_state: tuple=None,
        enc_outputs=None,
        **kwargs) -> tuple:
        '''forward'''
        wordvect = self.embedding(inputs)
        states = self.rnn(wordvect,training=training,initial_state=initial_state)
        outputs = states.pop(0)
        context_vector = self.attention([outputs,enc_outputs])
        outputs = self.concat([outputs,context_vector])
        outputs = self.dense(outputs)
        return (outputs,states)


class Seq2seq(keras.Model):

    def __init__(
        self,
        input_length=None,
        input_vocab_size=None,
        output_length=None,
        target_vocab_size=None,
        word_vect_size=8,
        recurrent_units=256,
        start_voc_id=0,
        **kwargs
    ):
        '''
        input_length: input sequence length
        input_vocab_size: vocabulary dictionary size for input sequence
        output_length: output sequence length
        target_vocab_size: vocabulary dictionary size for target sequence
        word_vect_size: word vector size of embedding layer
        recurrent_units: units of the recurrent layer
        dense_units: units of the full connection layer
        start_voc_id: vocabulary id of start word in input sequence
        '''
        super(Seq2seq, self).__init__(**kwargs)
        self.encoder = Encoder(
            input_length,
            input_vocab_size,
            word_vect_size,
            recurrent_units,
            **kwargs
        )
        self.decoder = Decoder(
            output_length,
            target_vocab_size,
            word_vect_size,
            recurrent_units,
            **kwargs
        )
        #self.out = keras.layers.Activation('softmax')
        self.start_voc_id = start_voc_id
        self.output_length = output_length

    def shiftSentence(
        self,
        sentence: ndarray,
        ) -> ndarray:
        '''shift target sequence to learn'''
        shape = tf.shape(sentence)
        batchs = shape[0]
        start_id = tf.expand_dims(tf.repeat([self.start_voc_id],repeats=[batchs]), 1)
        seq = sentence[:,:-1]
        result = tf.concat([start_id,seq],1)
        return result

    def call(
        self,
        inputs,
        training=None,
        mask=None,
        ):
        '''forward step'''
        train_data = inputs
        inputs,trues = train_data
        #print('--------forward step---------')
        #print(inputs)
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            print(train_data)
            raise Exception('error')
        enc_outputs,states = self.encoder(inputs,training)
        dec_inputs = self.shiftSentence(trues)

        outputs,dummy = self.decoder(dec_inputs,training,
            initial_state=states,enc_outputs=enc_outputs)
        #
        #outputs = self.out(outputs)
        return outputs

    def train_step(
        self,
        train_data: tuple
    ) -> dict:
        '''train step callback'''
        inputs,trues = train_data

        with tf.GradientTape() as tape:
            #print('====================-')
            #print(trues)
            outputs = self(train_data,training=True)
            loss = self.compiled_loss(
                trues,outputs,
                regularization_losses=self.losses)

        variables = self.trainable_variables

        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))

        self.compiled_metrics.update_state(trues, outputs)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """
        The logic for one evaluation step.
        """
        x, y = data

        y_pred = self(data, training=False)
        # Updates stateful loss metrics.
        self.compiled_loss(
            y, y_pred, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def translate(
        self,
        sentence: ndarray) -> ndarray:
        '''translate sequence'''
        input_length = sentence.size
        sentence = sentence.reshape([1,input_length])
        enc_outputs,states=self.encoder(sentence,training=True)
        voc_id = self.start_voc_id
        target_sentence =[]
        for i in range(self.output_length):
            inp = np.array([[voc_id]])
            predictions, states = self.decoder(inp,
                training=False,initial_state=states,enc_outputs=enc_outputs)
            voc_id = np.argmax(predictions)
            target_sentence.append(voc_id)

        return np.array(target_sentence)


# Generate the data
class NumAdditionDataset:

    def __init__(
            self,
            corpus_max: int,
            digits: int,
            reverse=True):
        self.vocab_input  = ['0','1','2','3','4','5','6','7','8','9','+',' ','@']
        self.vocab_target = ['0','1','2','3','4','5','6','7','8','9','+',' ','@']
        self.dict_input = dict(zip(self.vocab_input,range(len(self.vocab_input))))
        self.dict_target = dict(zip(self.vocab_target,range(len(self.vocab_target))))
        self.corpus_max = corpus_max
        self.digits = digits
        self.input_length = digits*2+1
        self.output_length = digits+1
        self.reverse = reverse

    def dicts(self) -> tuple:
        return (
            self.vocab_input,
            self.vocab_target,
            self.dict_input,
            self.dict_target,
        )

    def generate(self) -> tuple:
        '''generate random sequence'''
        max_num = pow(10,self.digits)
        max_sample = max_num ** 2 - 1
        numbers = np.random.choice(
            max_sample,max_sample,replace=False)
        questions = {}
        size = 0
        for i in range(max_sample):
            num = numbers[i]
            x1,x2 = sorted(divmod(num,max_num))
            question = '%d+%d' % (x1,x2)
            if questions.get(question) is not None:
                continue
            questions[question] = '%d' % (x1+x2)
            size += 1
            if size >= self.corpus_max:
                break
        numbers = None
        sequence = np.zeros([size,self.input_length],dtype=np.int32)
        target = np.zeros([size,self.output_length],dtype=np.int32)
        i = 0
        for question,answer in questions.items():
            question = question + " " * (self.input_length - len(question))
            answer = answer + " " * (self.output_length - len(answer))
            #if self.reverse:
            #    question = question[::-1]
            self.str2seq(
                question,
                self.dict_input,
                sequence[i])
            self.str2seq(
                answer,
                self.dict_target,
                target[i])
            i += 1
        return (sequence,target)

    def str2seq(
        self,
        input_string: str,
        word_dic: dict,
        output_seq: ndarray) -> None:
        '''translate string to sequence'''
        sseq = list(input_string.upper())
        seq_len = len(sseq)
        sp = word_dic[' ']
        bufsz=output_seq.size
        for i in range(bufsz):
            if i<seq_len:
                output_seq[i] = word_dic[sseq[i]]
            else:
                output_seq[i]=sp

    def seq2str(
        self,
        input_seq: ndarray,
        word_dic: dict
        ) -> str:
        '''translate string to sequence'''
        output_str = ''
        bufsz = input_seq.size
        for i in range(bufsz):
            output_str = output_str + word_dic[input_seq[i]]
        return output_str

    #def translate(
    #    self,
    #    model,
    #    input_str: str) -> str:
    #    '''translate sentence'''
    #    inputs = np.zeros([1,self.input_length],dtype=np.int32)
    #    self.str2seq(
    #        input_str,
    #        self.dict_input,
    #        inputs[0])
    #    outputs = model.predict(inputs)
    #    return self.seq2str(
    #        np.argmax(outputs[0],axis=1)),
    #        self.vocab_target)

    def load_data(
        self,
        path: str=None) -> ndarray:
        '''load dataset'''
        if path is None:
            path='numaddition-dataset.pkl'

        if os.path.exists(path):
            with open(path,'rb') as fp:
                dataset = pickle.load(fp)
        else:
            dataset = self.generate()
            with open(path,'wb') as fp:
                pickle.dump(dataset,fp)
        return dataset


epochs = 30;
batch_size = 128;
word_vect_size=256
recurrent_units=1024

print("embedding_dim: ",word_vect_size)
print("units: ",recurrent_units)

dataset = NumAdditionDataset(TRAINING_SIZE,DIGITS)

print("Generating data...")
questions,answers = dataset.load_data()
corpus_size = len(questions)
print("Total questions:", corpus_size)
input_voc,target_voc,input_dic,target_dic=dataset.dicts()

# Explicitly set apart 10% for validation data that we never train over.
split_at = len(questions) - len(questions) // 10
(x_train, x_val) = questions[:split_at], questions[split_at:]
(y_train, y_val) = answers[:split_at], answers[split_at:]

input_length  = x_train.shape[1] #DIGITS*2 + 1;
output_length = y_train.shape[1] #DIGITS + 1;
print("Input length =",input_length)
print("Output length=",output_length)


print("Training Data:")
print(x_train.shape)
print(y_train.shape)

print("Validation Data:")
print(x_val.shape)
print(y_val.shape)

#def create_seq2seq(
#    input_length=None,
#    input_vocab_size=None,
#    output_length=None,
#    target_vocab_size=None,
#    start_voc_id=None,
#    word_vect_size=None,
#    recurrent_units=None,
#    dense_units=None,
#):
#    def shiftSentence(
#        self,
#        sentence: ndarray,
#        start_voc_id,
#        ) -> ndarray:
#        '''shift target sequence to learn'''
#        shape = tf.shape(sentence)
#        batchs = shape[0]
#        start_id = tf.expand_dims(tf.repeat([start_voc_id],repeats=[batchs]), 1)
#        seq = sentence[:,:-1]
#        result = tf.concat([start_id,seq],1)
#        return result
#
#    encoder = Encoder(
#        input_length,
#        input_vocab_size,
#        word_vect_size,
#        recurrent_units,
#    )
#    decoder = Decoder(
#        output_length,
#        target_vocab_size,
#        word_vect_size,
#        recurrent_units,
#    )
#    #self.out = keras.layers.Activation('softmax')
#    #start_voc_id = start_voc_id
#    #output_length = output_length
#    inputs = keras.Input(shape=(None, input_length))
#
#    enc_outputs,states = encoder(inputs,training)
#    dec_inputs = shiftSentence(trues,start_voc_id)
#    outputs,dummy = decoder(dec_inputs,training,
#        initial_state=states,enc_outputs=enc_outputs)
#
#    model = tf.keras.Model([inputs,trues],trues)
#    return model
#
#seq2seq = create_seq2seq(
#    input_length=input_length,
#    input_vocab_size=len(input_dic),
#    output_length=output_length,
#    target_vocab_size=len(target_dic),
#    start_voc_id=dataset.dict_target['@'],
#    word_vect_size=16,
#    recurrent_units=128,
#    dense_units=128,
#)


seq2seq = Seq2seq(
    input_length=input_length,
    input_vocab_size=len(input_dic),
    output_length=output_length,
    target_vocab_size=len(target_dic),
    start_voc_id=dataset.dict_target['@'],
    word_vect_size=word_vect_size,
    recurrent_units=recurrent_units,
)




print("Compile model...")
seq2seq.compile(
    #loss='sparse_categorical_crossentropy',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'],
    )

print("Train model...")
history = seq2seq.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
)

for i in range(10):
    idx = np.random.randint(0,len(questions))
    question = questions[idx]
    input = question.reshape(1,input_length)

    #input = keras.utils.to_categorical(
    #    input.reshape(input.size,),
    #    num_classes=len(input_voc)
    #    ).reshape(input.shape[0],input.shape[1],len(input_voc))

    #predict = model.predict(input)
    #predict_seq = np.argmax(predict[0].reshape(output_length,len(target_dic)),axis=1)
    predict_seq = seq2seq.translate(question);

    predict_str = dataset.seq2str(predict_seq,target_voc)
    question_str = dataset.seq2str(question,input_voc)
    answer_str = dataset.seq2str(answers[idx],target_voc)
    correct = '*' if predict_str==answer_str else ' '
    print('%s=%s : %s %s' % (question_str,predict_str,correct,answer_str))

plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_loss'],label='val_loss')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.legend()
plt.show()
