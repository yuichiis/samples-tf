# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

tok = tf.keras.preprocessing.text.Tokenizer(
    num_words=8,filters="",
)
x = [
    "Hello.\n",
    "I am Tom.\n",
    "How are you?\n",
    "Hello Tom.\n",
    "I am fine.\n",
    "I am Jerry.\n",
    "How are you?\n",
    "I am fine too\n",
]
tok.fit_on_texts(x)
print(tok.index_word)
seq = tok.texts_to_sequences(x)
print(tok.sequences_to_texts(seq))
