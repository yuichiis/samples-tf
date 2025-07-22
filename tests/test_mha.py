import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, Dense, Embedding, Input, LayerNormalization
from tensorflow.keras.models import Model
import numpy as np

# Define the MultiHeadAttention layer with causal masking
class CausalMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim):
        super(CausalMultiHeadAttention, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.layernorm = LayerNormalization()

    def call(self, query, value):
        # Create a causal mask
        seq_len = tf.shape(query)[1]
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        attention_output = self.mha(query, value, attention_mask=causal_mask)
        return self.layernorm(attention_output + query)

# Define the Seq2Seq model
class Seq2SeqModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_heads, key_dim):
        super(Seq2SeqModel, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.encoder_mha = CausalMultiHeadAttention(num_heads, key_dim)
        self.decoder_mha = CausalMultiHeadAttention(num_heads, key_dim)
        self.dense = Dense(vocab_size)

    def call(self, encoder_input, decoder_input):
        encoder_embedded = self.embedding(encoder_input)
        decoder_embedded = self.embedding(decoder_input)
        encoder_output = self.encoder_mha(encoder_embedded, encoder_embedded)
        decoder_output = self.decoder_mha(decoder_embedded, encoder_output)
        return self.dense(decoder_output)

# Hyperparameters
vocab_size = 100
embedding_dim = 64
num_heads = 4
key_dim = 64

# Create the model
model = Seq2SeqModel(vocab_size, embedding_dim, num_heads, key_dim)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Dummy data for training
encoder_input_data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
decoder_input_data = np.array([[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]])
decoder_output_data = np.array([[12, 13, 14, 15, 16], [17, 18, 19, 20, 21]])

# Train the model
model.fit([encoder_input_data, decoder_input_data], decoder_output_data, epochs=10)

# Predict
test_encoder_input = np.array([[1, 2, 3, 4, 5]])
test_decoder_input = np.array([[11]])
prediction = model.predict([test_encoder_input, test_decoder_input])
print("Prediction:", np.argmax(prediction, axis=-1))