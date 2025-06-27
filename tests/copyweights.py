import tensorflow as tf
import numpy as np  

class MyModel(tf.keras.Model):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.input_check = tf.keras.layers.Input(shape=[1])
        self.dense_layer0 = tf.keras.layers.Dense(
            units=1,
        )
        

    #@tf.function(input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float32)])
    def __call__(self, x):
        x = self.dense_layer0(x)
        return x


#model = tf.keras.Sequential([
#    tf.keras.layers.Dense(
#        input_shape=[1],
#        units=1, #kernel_initializer='ones', bias_initializer='zeros'
#    ),
#])
#loaded_model = tf.keras.Sequential([
#    tf.keras.layers.Dense(
#        input_shape=[1],
#        units=1, #kernel_initializer='ones', bias_initializer='zeros'
#    ),
#])

# Create some dummy data
x = tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32)

# Create an instance of the model
model = MyModel(name="my_model")
model.compile()
#y = model(x)
#
## Load the model
loaded_model = MyModel(name="loaded_model")
loaded_model.compile()
#loaded_y = loaded_model(x)

# Copy the weights
weights = model.get_weights()
print(weights)
loaded_model.set_weights(weights)
print(loaded_model.get_weights())


# Call the model to build the graph
y = model(x)

# Call the model to build the graph
y = model(x)

# Test the loaded model
loaded_y = loaded_model(x)

# Print the results
print("Original Model Output:")
#print(y)
print("Loaded Model Output:")
print(loaded_y)

# Assert that the outputs are the same
np.testing.assert_array_equal(y.numpy(), loaded_y.numpy())

print("Model saved and loaded successfully!")
