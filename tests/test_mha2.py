import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, Dense, Embedding, Input, LayerNormalization
from tensorflow.keras.models import Model
import numpy as np

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
  def __init__(self, vocab_size, d_model, **kwargs):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(
        vocab_size, d_model,
        **kwargs,
    )
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

# Define the Seq2Seq model
class AttnModel(tf.keras.Model):
    def __init__(self, vocab_size, d_model, num_heads, key_dim):
        super(AttnModel, self).__init__()
        self.enc_emb = PositionalEmbedding(vocab_size, d_model, mask_zero=True)
        self.enc_mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.dec_emb = PositionalEmbedding(vocab_size, d_model, mask_zero=True)
        self.dec_mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.cross_mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, inputs):
        [inputs, targets] = inputs
        inputs = self.enc_emb(inputs)
        inputs =self.enc_mha(inputs, inputs, inputs)
        targets = self.dec_emb(targets)
        targets = self.dec_mha(targets, targets, targets, use_causal_mask=True)
        cross = self.cross_mha(targets, inputs, inputs)
        return cross
        

# Hyperparameters
num_heads = 4
key_dim = 64
vocab_size = 20
d_model = 64

# Create the model
model = AttnModel(vocab_size, d_model, num_heads, key_dim)

# Compile the model
lossfn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=lossfn, metrics=['accuracy'])

# Dummy data for training
inputs = np.array([
    [10,2,3,4,11,0,0,0,],
    [10,4,3,2,1,11,0,0,],
])
targets = np.array([
    [10,1,2,1,2,3,11,0,],
    [10,3,4,3,4,1,11,0,],
])
labels = np.array([
    [1,2,1,2,3,11,0,0,],
    [3,4,3,4,1,11,0,0,],
])

# Train the model
model.fit([inputs,targets], labels, epochs=1000, verbose=0)

# Predict
test_inputs = np.array([
    [10,2,3,4,11,0,0,0,],
])
test_targets = np.array([
    [10,1,2,1,2,3,11,0,],
])
preds = model.predict([test_inputs, test_targets])
#print("Prediction:", prediction)
print("Inputs:",test_inputs.shape)
print("targets:",test_targets.shape)
print("Preds :",preds.shape)
print(np.argmax(preds, axis=-1))
