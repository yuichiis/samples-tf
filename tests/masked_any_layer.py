import tensorflow as tf

input_tensor = tf.constant([
    [1, 2, 0, 4, 0],
    [1, 2, 0, 4, 0],
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

class ApplyMask(tf.keras.layers.Layer):
    def __init__(self, layername=None):
        self.layername = layername
        super(ApplyMask, self).__init__()

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
            #inputs = inputs*tf.expand_dims(tf.cast(mask,tf.float32),axis=-1)
        else:
            print("No mask (in "+self.layername+".call)")
        return inputs


##############################################################################
emb = tf.keras.layers.Embedding(10, 4, mask_zero=True, input_length=5)
customFirst = MyCustomLayer(layername='First')
customSecond = MyCustomLayer(layername='Second')
customThird = MyCustomLayer(layername='Third')
applyMask = ApplyMask(layername='ApplyMask')
lstm = tf.keras.layers.LSTM(8, return_sequences=True)
dense = tf.keras.layers.Dense(10)
mha = tf.keras.layers.MultiHeadAttention(8,8)
drop = tf.keras.layers.Dropout(0.5)
softmax = tf.keras.layers.Softmax()
norm = tf.keras.layers.LayerNormalization()
masking = tf.keras.layers.Masking(mask_value=0)


#print('dense has compute_mask :', hasattr(dense, 'compute_mask'))
#print('lstm has compute_mask :', hasattr(lstm, 'compute_mask'))
#print('mha has compute_mask :', hasattr(mha, 'compute_mask'))
#print('drop has compute_mask :', hasattr(drop, 'compute_mask'))
#print('softmax has compute_mask :', hasattr(softmax, 'compute_mask'))

x = tf.random.normal((2,5,4),dtype=tf.float32)
mask = tf.cast(input_tensor,tf.bool)
#x = emb(input_tensor)   # (1,5,4) <= (1,5) 

print('x=',x)
x = applyMask(x,mask)
#x = masking(x)          # (1,5,4) <= (1,5,4) 
print('x=',x)
x = customFirst(x)      # (1,5,4) <= (1,5,4)
#x = dense(x)
#x = lstm(x)
#x = x * 2
#x = drop(x)
#x = softmax(x)
#x = norm(x)
x = customSecond(x)     # (1,5,4) <= (1,5,4)
#output = x
#output = lstm(x)        # (1,5,4) <= (1,5,8)
#x = customThird(x)      # (1,5,4) <= (1,5,4)
output = x

print('input',input_tensor)
print('output',output)
#print(output.shape)
