import tensorflow as tf

input_tensor = tf.constant([
    [1, 2, 0, 4, 0],
    [1, 2, 0, 4, 0],
])
input_tensor1 = tf.constant([
    [1, 2, 0, 4, 5, 0],
    [1, 2, 0, 4, 5, 0],
])
input_tensor2 = tf.constant([
    [1, 0, 3, 0, 0, 6],
    [1, 0, 3, 0, 0, 6],
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
emb  = tf.keras.layers.Embedding(10, 4, mask_zero=True,  input_length=5)
emb1 = tf.keras.layers.Embedding(10, 4, mask_zero=True, input_length=6)
emb2 = tf.keras.layers.Embedding(10, 4, mask_zero=True, input_length=6)
customFirst = MyCustomLayer(layername='First')
customSecond = MyCustomLayer(layername='Second')
customThird = MyCustomLayer(layername='Third')
applyMask = ApplyMask(layername='ApplyMask')
lstm = tf.keras.layers.LSTM(8, return_sequences=True)
dense = tf.keras.layers.Dense(10)
mha = tf.keras.layers.MultiHeadAttention(8,4)
drop = tf.keras.layers.Dropout(0.5)
softmax = tf.keras.layers.Softmax()
add = tf.keras.layers.Add()
norm = tf.keras.layers.LayerNormalization()
masking = tf.keras.layers.Masking(mask_value=0)


#print('dense has compute_mask :', hasattr(dense, 'compute_mask'))
#print('lstm has compute_mask :', hasattr(lstm, 'compute_mask'))
#print('mha has compute_mask :', hasattr(mha, 'compute_mask'))
#print('drop has compute_mask :', hasattr(drop, 'compute_mask'))
#print('softmax has compute_mask :', hasattr(softmax, 'compute_mask'))

x = tf.random.normal((2,5,4),dtype=tf.float32)
y = tf.random.normal((2,5),dtype=tf.float32)
mask = tf.cast(input_tensor,tf.bool)
mask_y = [[True,True,True,False,False],[True,True,True,False,False]]
#x = emb(input_tensor)       # (2,5,4) <= (2,5) 
#x1 = emb1(input_tensor1)   # (2,6,4) <= (2,6) 
#x2 = emb2(input_tensor2)   # (2,6,4) <= (2,6)

#print('x=',x)
x = applyMask(x,mask)
y = applyMask(y,mask_y)
#x = masking(x)          # (1,5,4) <= (1,5,4) 
#print('x=',x)
#x = customFirst(x)      # (1,5,4) <= (1,5,4)
#x1 = customFirst(x1)      # (1,6,4) <= (1,6,4)
#x2 = customFirst(x2)      # (1,6,4) <= (1,6,4)
#x = dense(x)
#x = lstm(x)
#[x,scores] = mha(x,x1,x2,return_attention_scores=True)
#x = x * 2
#x = drop(x)
#x = softmax(x)
#x = norm(x)
#x = add([x1,x2])
x = add([x,y])
#x = customSecond(x)     # (1,5,4) <= (1,5,4)
x = customSecond(x)     # (1,6,4) <= (1,6,4)
#scores = customThird(scores)     # (1,5,4) <= (1,5,4)
#output = x
#output = lstm(x)        # (1,5,4) <= (1,5,8)
#x = customThird(x)      # (1,5,4) <= (1,5,4)
output = x

#print('input',input_tensor)
#print('output',output)
#print(output.shape)
