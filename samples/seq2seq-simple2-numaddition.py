# coding: utf-8

import os
import pickle
import numpy as np
from numpy import ndarray
import tensorflow as tf
from tensorflow import keras
from zipfile import ZipFile

batch_size = 32  # Batch size for training.
epochs = 30  # Number of epochs to train for.
latent_dim = 128  # Latent dimensionality of the encoding space.
# Path to the data txt file on disk.
#data_path = "fra.txt"
# Vectorize the data.

# Parameters for the model and dataset.
#TRAINING_SIZE = 50000
TRAINING_SIZE = 5000
DIGITS = 3
REVERSE = True

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

input_length  = DIGITS*2 + 1
output_length = DIGITS + 1

dataset = NumAdditionDataset(TRAINING_SIZE,DIGITS,REVERSE)

print("Generating data...")
questions,answers = dataset.load_data()
corpus_size = len(questions)
print("Total questions:", corpus_size)
input_voc,target_voc,input_dic,target_dic=dataset.dicts()



# Explicitly set apart 10% for validation data that we never train over.
#split_at = len(questions) - len(questions) // 10
#(x_train, x_val) = questions[:split_at], questions[split_at:]
#(y_train, y_val) = answers[:split_at], answers[split_at:]

#print("Training Data:")
#print(x_train.shape)
#print(y_train.shape)
#
#print("Validation Data:")
#print(x_val.shape)
#print(y_val.shape)

decoder_inputs_data = np.zeros_like(answers)
decoder_inputs_data[:,0] = target_dic['@']
for t in range(answers.shape[1]-1):
    decoder_inputs_data[:,t+1] = answers[:,t]

encoder_inputs_data = keras.utils.to_categorical(
    questions.reshape(questions.size,),
    num_classes=len(input_voc)
    ).reshape(questions.shape[0],questions.shape[1],len(input_voc))
decoder_inputs_data = keras.utils.to_categorical(
    decoder_inputs_data.reshape(decoder_inputs_data.size,),
    num_classes=len(target_voc)
    ).reshape(decoder_inputs_data.shape[0],decoder_inputs_data.shape[1],len(target_voc))
decoder_target_data = keras.utils.to_categorical(
    answers.reshape(answers.size,),
    num_classes=len(target_voc)
    ).reshape(answers.shape[0],answers.shape[1],len(target_voc))

num_encoder_tokens = len(input_voc)
num_decoder_tokens = len(target_voc)

# Define an input sequence and process it.
encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
#encoder = keras.layers.LSTM(latent_dim, return_state=True)
#encoder_outputs, state_h, state_c = encoder(encoder_inputs)
#
## We discard `encoder_outputs` and only keep the states.
#encoder_states = [state_h, state_c]

encoder = keras.layers.GRU(latent_dim, return_state=True)
encoder_outputs, state_h = encoder(encoder_inputs)
encoder_states = [state_h]


# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
#decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
#decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder = keras.layers.GRU(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _ = decoder(decoder_inputs, initial_state=encoder_states)
decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`

model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.summary()





model.fit(
    [encoder_inputs_data, decoder_inputs_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)
# Save model
model.save("s2s")

############################################################################

# Define sampling models
# Restore the model and construct the encoder and decoder.
model = keras.models.load_model("s2s")

encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = keras.Input(shape=(latent_dim,), name="input_3")
decoder_state_input_c = keras.Input(shape=(latent_dim,), name="input_4")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]
    return decoded_sentence

for seq_index in range(20):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index : seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print("-")
    print("Input sentence:", input_texts[seq_index])
    print("Decoded sentence:", decoded_sentence)
