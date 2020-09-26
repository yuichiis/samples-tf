# coding: utf-8

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from zipfile import ZipFile

# Download the file
path_to_zip = 'fra-eng.zip'
extract_to = os.path.dirname(os.path.abspath(path_to_zip))+"/fra-eng/"
data_path = extract_to+"fra.txt"

with ZipFile(path_to_zip) as f:
    f.extractall(path='fra-eng/')

print(data_path)
batch_size = 64  # Batch size for training.
epochs = 100#100  # Number of epochs to train for.
latent_dim = 256#256  # Latent dimensionality of the encoding space.
embedding_dim = 128
num_samples = 10000#10000  # Number of samples to train on.
# Path to the data txt file on disk.
#data_path = "fra.txt"
# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
for line in lines[: min(num_samples, len(lines) - 1)]:
    target_text, input_text, _ = line.split("\t")
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = "\t" + target_text + "\n"
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    #decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0

#encoder_input_sequence = np.full(
#    (len(input_texts), max_encoder_seq_length), input_token_index[" "],
#    dtype=np.int32,
#)
#decoder_target_sequence = np.full(
#    (len(target_texts), max_decoder_seq_length), target_token_index[" "],
#    dtype=np.int32,
#)
#for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
#    for t, char in enumerate(input_text):
#        encoder_input_sequence[i, t] = input_token_index[char]
#    for t, char in enumerate(target_text):
#        decoder_target_sequence[i, t] = target_token_index[char]


# Define an input sequence and process it.
model = keras.Sequential([
    keras.Input(shape=(None, num_encoder_tokens)),
    #keras.layers.Embedding(num_encoder_tokens,embedding_dim),
    keras.layers.GRU(latent_dim),
    keras.layers.RepeatVector(max_decoder_seq_length),
    keras.layers.GRU(latent_dim, return_sequences=True),
    keras.layers.Dense(num_decoder_tokens, activation="softmax"),
])
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    #optimizer="adam",
    #loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #metrics=["accuracy"]
)
model.summary()
model.fit(
    #encoder_input_sequence,
    #decoder_target_sequence,
    encoder_input_data,
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

#encoder_inputs = model.input[0]  # input_1
#encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
#encoder_states = [state_h_enc, state_c_enc]
#encoder_model = keras.Model(encoder_inputs, encoder_states)
#
#decoder_inputs = model.input[1]  # input_2
#decoder_state_input_h = keras.Input(shape=(latent_dim,), name="input_3")
#decoder_state_input_c = keras.Input(shape=(latent_dim,), name="input_4")
#decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
#decoder_lstm = model.layers[3]
#decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
#    decoder_inputs, initial_state=decoder_states_inputs
#)
#decoder_states = [state_h_dec, state_c_dec]
#decoder_dense = model.layers[4]
#decoder_outputs = decoder_dense(decoder_outputs)
#decoder_model = keras.Model(
#    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
#)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(model,input_seq):
    # Encode the input as state vectors.
    states_value = model.predict(input_seq)
    target_seq = np.argmax(states_value[0],axis=1)
    target_text = ''
    for c in target_seq:
        sampled_char = reverse_target_char_index[c]
        target_text += sampled_char
    return target_text
    # Generate empty target sequence of length 1.
    #target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    #target_seq[0, 0, target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    #stop_condition = False
    #decoded_sentence = ""
    #while not stop_condition:
    #    output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
    #
    #    # Sample a token
    #    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    #    sampled_char = reverse_target_char_index[sampled_token_index]
    #    decoded_sentence += sampled_char
    #
    #    # Exit condition: either hit max length
    #    # or find stop character.
    #    if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
    #        stop_condition = True

    #    # Update the target sequence (of length 1).
    #    target_seq = np.zeros((1, 1, num_decoder_tokens))
    #    target_seq[0, 0, sampled_token_index] = 1.0

    #    # Update states
    #    states_value = [h, c]
    #return decoded_sentence

seq_index = 0
prev_input = None
for n in range(20):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_sequence[seq_index : seq_index + 1]
    while input_texts[seq_index] == prev_input:
        seq_index += 1
    input_seq = encoder_input_sequence[seq_index : seq_index + 1]
    decoded_sentence = decode_sequence(model,input_seq)
    print("-")
    print("Input sentence:", input_texts[seq_index])
    print("Decoded sentence:", decoded_sentence)
    print("Correct sentence:", target_texts[seq_index])
    prev_input = input_texts[seq_index]
    seq_index += 1
