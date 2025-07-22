import tensorflow as tf
import numpy as np  

class MyModel(tf.keras.Model):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.dense_layer0 = tf.keras.layers.Dense(
            input_shape=[1],
            units=1, #kernel_initializer='ones', bias_initializer='zeros'
        )
        self.dense_layer1 = tf.keras.layers.Dense(
            units=1, #kernel_initializer='ones', bias_initializer='zeros'
        )

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float32)])
    def __call__(self, x):
        x = self.dense_layer0(x)
        x = self.dense_layer1(x)
        return x

# Create an instance of the model
model = MyModel(name="my_model")
#model.compile()


# Create some dummy data
x = tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32)

# Call the model to build the graph
y = model(x)

# Save the model
#tf.saved_model.save(model, "saved_model")
model.save_weights("saved_model")

# Load the model
#loaded_model = tf.saved_model.load("saved_model")
loaded_model = MyModel(name="my_model")
loaded_model.load_weights("saved_model")

# Call the model to build the graph
y = model(x)
# Test the loaded model
loaded_y = loaded_model(x)

# Print the results
print("Original Model Output:")
print(y)
print("Loaded Model Output:")
print(loaded_y)

# Assert that the outputs are the same
np.testing.assert_array_equal(y.numpy(), loaded_y.numpy())

print("Model saved and loaded successfully!")
    