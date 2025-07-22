import tensorflow as tf

input_tensor = tf.constant([
    [1, 2, 0, 4, 0],
])


class MyCustomLayer(tf.keras.layers.Layer):
    def __init__(self, layername=None):
        self.layername = layername
        super(MyCustomLayer, self).__init__()

    def compute_mask(self, inputs, mask=None):
        #print(inputs)
        if mask is not None:
            print("Input mask (retrieved in "+self.layername+".compute_mask):", mask)
            mask = tf.math.logical_not(mask)
        else:
            print("No mask (in "+self.layername+".compute_mask)")
        return mask

    def call(self, inputs, mask=None):

        if mask is not None:
            print("Original mask (retrieved in "+self.layername+".call):", mask)
            inputs += 1
        else:
            print("No mask (in "+self.layername+".call)")
        return inputs

##############################################################################
#model = keras.Sequential([
#    tf.keras.layers.Embedding(10, 4, mask_zero=True, input_length=5),
#    MyCustomLayer(layername='First'),
#    MyCustomLayer(layername='Second'),
#    tf.keras.layers.LSTM(8)
#])
# output = model(input_tensor)

##############################################################################
emb = tf.keras.layers.Embedding(10, 4, mask_zero=True, input_length=5)
customFirst = MyCustomLayer(layername='First')
customSecond = MyCustomLayer(layername='Second')
lstm = tf.keras.layers.LSTM(8)

x = emb(input_tensor)
x = customFirst(x)
x = x * 2
x = customSecond(x)
#output = lstm(x)
output = x

##############################################################################
#class MyModel1(tf.keras.Model):
#    def __init__(self):
#        super().__init__()
#        self.emb = tf.keras.layers.Embedding(10, 4, mask_zero=True, input_length=5)
#        self.customFirst = MyCustomLayer(layername='First')
#
#    def call(self, x):
#        x = self.emb(x)
#        x = self.customFirst(x)
#        return x
#
#class MyModel2(tf.keras.Model):
#    def __init__(self):
#        super().__init__()
#        self.customSecond = MyCustomLayer(layername='Second')
#        self.lstm = tf.keras.layers.LSTM(8)
#
#    def call(self, x):
#        x = self.customSecond(x)
#        x = self.lstm(x)
#        return x
#
#model1 = MyModel1()
#model2 = MyModel2()
#x = model1(input_tensor)
#output = model2(x)

print(output)
