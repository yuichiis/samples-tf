import tensorflow as tf

input_query_tensor = tf.constant([
    [1, 2, 3, 4, 0],
    [1, 2, 3, 4, 0],
])

input_value_tensor = tf.constant([
    [1, 2, 3, 0, 0],
    [1, 2, 3, 0, 0],
])

input_key_tensor = tf.constant([
    [1, 2, 0, 0, 0],
    [1, 2, 0, 0, 0],
])


class MyCustomLayer(tf.keras.layers.Layer):
    def __init__(self, layername=None):
        self.layername = layername
        super(MyCustomLayer, self).__init__()

    def compute_mask(self, inputs, mask=None):
        #print(inputs)
        if mask is not None:
            print("Input ", inputs.shape, ", mask (retrieved in "+self.layername+".compute_mask):", mask)
        else:
            print("No mask (in "+self.layername+".compute_mask)")
        return mask

    def call(self, inputs, mask=None):

        if mask is not None:
            print("Input ", inputs.shape, ", Original mask (retrieved in "+self.layername+".call):", mask)
            inputs += 1
        else:
            print("No mask (in "+self.layername+".call)")
        return inputs

class MyCustomLayer0(tf.keras.layers.Layer):
    def __init__(self, layername=None):
        self.layername = layername
        super(MyCustomLayer0, self).__init__()

    def call(self, inputs, mask=None):

        if mask is not None:
            print("Input ", inputs.shape, ", Original mask (retrieved in "+self.layername+".call):", mask)
            inputs += 1
        else:
            print("No mask (in "+self.layername+".call)")
        return inputs

class MyCustomLayer2(tf.keras.layers.Layer):
    def __init__(self, layername=None):
        self.layername = layername
        super(MyCustomLayer2, self).__init__()

    def compute_mask(self, inputs, mask=None):
        #print(inputs)
        if mask is not None:
            print("Input ", inputs.shape, ", mask (retrieved in "+self.layername+".compute_mask):", mask)
            mask = tf.expand_dims(mask, axis=-1)
        else:
            print("No mask (in "+self.layername+".compute_mask)")
        return mask

    def call(self, inputs, mask=None):

        if mask is not None:
            print("Input ", inputs.shape, ", Original mask (retrieved in "+self.layername+".call):", mask)
            inputs += 1
        else:
            print("No mask (in "+self.layername+".call)")
        return inputs

##############################################################################
emb0 = tf.keras.layers.Embedding(10, 4, mask_zero=True, input_length=5)
emb1 = tf.keras.layers.Embedding(10, 4, mask_zero=True, input_length=5)
emb2 = tf.keras.layers.Embedding(10, 4, mask_zero=True, input_length=5)
customFirst0 = MyCustomLayer(layername='First0')
customFirst1 = MyCustomLayer(layername='First1')
customFirst2 = MyCustomLayer(layername='First2')
customSecond0 = MyCustomLayer(layername='Second0')
customSecond1 = MyCustomLayer(layername='Second1')
customThird0 = MyCustomLayer(layername='Third0')
customThird1 = MyCustomLayer(layername='Third1')
attention = tf.keras.layers.Attention()

print('attention has compute_mask :', hasattr(attention, 'compute_mask'))

q = emb0(input_query_tensor)   # (1,5,4) <= (1,5) 
v = emb1(input_value_tensor)   # (1,5,4) <= (1,5) 
k = emb2(input_key_tensor)   # (1,5,4) <= (1,5) 
q = customFirst0(q)      # (1,5,4) <= (1,5,4)
v = customFirst1(v)      # (1,5,4) <= (1,5,4)
k = customFirst2(k)      # (1,5,4) <= (1,5,4)
#x = dense(x)
#x = lstm(x)
#x = drop(x)
#x = softmax(x)
#q = customSecond0(q)     # (1,5,4) <= (1,5,4)
#v = customSecond1(v)     # (1,5,4) <= (1,5,4)
#output = x
output,scores = attention([q,v,k],return_attention_scores=True)        # (1,5,4) <= (1,5,8)
outputz = customThird0(output)      # (1,5,4) <= (1,5,4)
scoresz = customThird1(scores)      # (1,5,4) <= (1,5,4)

#print(output)
print(output.shape)
