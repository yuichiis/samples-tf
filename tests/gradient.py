import tensorflow as tf
import numpy as np

#x = tf.Variable(2.)
#y = tf.Variable(3.)
#with tf.GradientTape(persistent=True) as tape:
#    out = x * y
#print(tape.gradient(out,x))
#print(tape.gradient(out,y))
## 3
## 2
#with tf.GradientTape() as tape:
#    out = x * y
#print(tape.gradient(out,x))
## 3


######################################
#x = tf.Variable(2.)
#with tf.GradientTape(persistent=True) as tape:
#    print('in')
#    y = x * x
#    z = y * y
#print('out')
#print(tape.gradient(z,x))
#print(tape.gradient(y,x))
#print(tape.gradient(z,y))

######################################
#layer = tf.keras.layers.Dense(units=5,input_shape=[1])
##layer2 = tf.keras.layers.Dense(units=5)
#x = tf.Variable(np.array([[3.0],[4.0]]))
#with tf.GradientTape() as tape:
#    outputs = layer(x)
#
#gradients = tape.gradient(outputs,layer.weights)
#print('layer')
#print(gradients)

######################################
#x = tf.constant(2.)
#with tf.GradientTape(persistent=True) as tape:
#    tape.watch(x)
#    y = x * x
#print('watch')
#print(tape.gradient(y,x))


######################################
#class TestModel(tf.keras.Model):
#    def __init__(self,**kwargs):
#        super(TestModel, self).__init__(**kwargs)
#        self.layer = tf.keras.layers.Dense(4,input_shape=(3,))
#
#    def call(self,inputs):
#        outputs = self.layer(inputs)
#        return outputs
#
#model = TestModel()
#
#inputs = np.array([[2.0,2.0,2.0],[3.0,3.0,3.0]])
#with tf.GradientTape() as tape:
#    outputs = model(inputs)
#gradients = tape.gradient(outputs, model.trainable_variables)
#print('model')
#print(gradients)

#loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#optimizer = tf.keras.optimizers.Adam()
#trains = np.zeros([10,6])
#labels = np.zeros([10],dtype=np.int32)
#ds = tf.data.Dataset.from_tensor_slices((trains,labels)).batch(2)
#for inputs,trues in ds:
#    with tf.GradientTape() as tape:
#        outputs = model(inputs)
#        loss = loss_function(trues,outputs)
#    grads = tape.gradient(loss,model.trainable_variables)
#    optimizer.apply_gradients(zip(grads, model.trainable_variables))

##################################################
#x = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]],dtype=np.float32)
#x = tf.Variable(x)
#trues = np.array([1, 2],dtype=np.int32)
#activation = tf.nn.softmax
#loss_function = tf.keras.losses.SparseCategoricalCrossentropy()#from_logits=True
#with tf.GradientTape() as tape:
#    print('x=',x)
#    xx = activation(x,axis=-1)
#    print('xx=',xx)
#    loss = loss_function(trues,xx)
#
#gradients = tape.gradient(loss, x)
#print('sparse')
#print('loss=',loss)
#print('gradients=',gradients)

##################################################
#x = tf.Variable(np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]],dtype=np.float32))
#trues = np.array([[0, 1, 0], [0, 0, 1]],dtype=np.float32)
#
#activation = tf.nn.softmax
#loss_function = tf.keras.losses.CategoricalCrossentropy()#from_logits=True
#with tf.GradientTape() as tape:
#    xx = activation(x)
#    loss = loss_function(trues,xx)
#
#gradients = tape.gradient(loss, x)
#print('categorical')
#print(loss)
#print(gradients)



##################################################
#x = tf.Variable(np.array([-18.6, 0.51, 2.94, -12.8],dtype=np.float32))
#trues = np.array([0, 1, 0, 0],dtype=np.float32)
#x = tf.Variable(np.array([-2.0, 2.0, 0.0],dtype=np.float32))
#trues = np.array([0.0, 1.0 , 0.0],dtype=np.float32)
#
##activation = tf.nn.softmax
#loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#with tf.GradientTape() as tape:
#    #xx = activation(x)
#    loss = loss_function(trues,x)
#
#gradients = tape.gradient(loss, x)
#print('binarycrossentropy')
#print(loss)
#print(gradients)


##################################################
#x = tf.Variable(np.array([[0., 1.], [0., 0.]],dtype=np.float32))
#trues = np.array([[1., 1.], [1., 0.]],dtype=np.float32)
#
##activation = tf.nn.softmax
#loss_function = tf.keras.losses.MeanSquaredError()
#with tf.GradientTape() as tape:
#    #xx = activation(x)
#    loss = loss_function(trues,x)
#
#gradients = tape.gradient(loss, x)
#print('mse')
#print(loss)
#print(gradients)

##################################################
#a = tf.Variable(np.array([[1, 2, 3], [4, 5, 6]],dtype=np.float32))
#b = tf.Variable(np.array([[7,8],[9,10],[11,12]],dtype=np.float32))
#
#with tf.GradientTape() as tape:
#    c = tf.matmul(a,b)
#
#gradients = tape.gradient(c, [a,b])
#print('matmul')
#print(c)
#print(gradients)

##################################################
#a = tf.Variable(np.array([[1,4],[2,5],[3,6]],dtype=np.float32))
#b = tf.Variable(np.array([[7,8],[9,10],[11,12]],dtype=np.float32))
#
#with tf.GradientTape() as tape:
#    c = tf.matmul(a,b,transpose_a=True)
#
#gradients = tape.gradient(c, [a,b])
#print('matmul_transposeA')
#print(c)
#print(gradients)

##################################################
#a = tf.Variable(np.array([[1, 2, 3], [4, 5, 6]],dtype=np.float32))
#b = tf.Variable(np.array([[7,9,11],[8,10,12]],dtype=np.float32))
#
#with tf.GradientTape() as tape:
#    c = tf.matmul(a,b,transpose_b=True)
#
#gradients = tape.gradient(c, [a,b])
#print('matmul_transposeB')
#print(c)
#print(gradients)

##################################################
#a = tf.Variable(np.arange(1, 13, dtype=np.float32).reshape([2, 2, 3]))
#b = tf.Variable(np.arange(13, 25, dtype=np.float32).reshape([2, 3, 2]))
#
#with tf.GradientTape() as tape:
#    c = tf.matmul(a,b)
#
#gradients = tape.gradient(c, [a,b])
#print('matmul_batch')
#print('c=',c)
#print('grads=',gradients)

##################################################
#a = tf.Variable(np.arange(1, 13, dtype=np.float32).reshape([2, 3, 2]))
#b = tf.Variable(np.arange(13, 25, dtype=np.float32).reshape([2, 3, 2]))
#
#with tf.GradientTape() as tape:
#    c = tf.matmul(a,b,transpose_a=True)
#
#gradients = tape.gradient(c, [a,b])
#print('matmul_batch_transposeA')
#print('c=',c)
#print('grads=',gradients)

##################################################
#a = tf.Variable(np.arange(1, 13, dtype=np.float32).reshape([2, 2, 3]))
#b = tf.Variable(np.arange(13, 25, dtype=np.float32).reshape([2, 2, 3]))
#
#with tf.GradientTape() as tape:
#    c = tf.matmul(a,b,transpose_b=True)
#
#gradients = tape.gradient(c, [a,b])
#print('matmul_batch_transposeB')
#print('c=',c)
#print('grads=',gradients)

##################################################
#a = tf.Variable(np.arange(1, 25, dtype=np.float32).reshape([2, 3, 4]))
#print('transpose')
#print('a=',a)
#print('transpose_a=',tf.transpose(a))

##################################################
#a = tf.Variable(np.arange(1, 37, dtype=np.float32).reshape([2,3,2,3]))
#b = tf.Variable(np.arange(37, 73, dtype=np.float32).reshape([2,3,2,3]))
#
#with tf.GradientTape() as tape:
#    c = tf.matmul(a,b,transpose_a=True)
#
#gradients = tape.gradient(c, [a,b])
#print('matmul_4d_batch_transposeA')
#print('c=',c)
#print('grads=',gradients)

##################################################
#a = tf.Variable(np.array([[1, 2, 3], [4, 5, 6]],dtype=np.float32))
#b = tf.Variable(np.array([[7,8],[9,10],[11,12]],dtype=np.float32))
#
#with tf.GradientTape() as tape:
#    c = tf.matmul(a,b,transpose_a=True,transpose_b=True)
#
#gradients = tape.gradient(c, [a,b])
#print('matmul transposeAB')
#print(c)
#print(gradients)

##################################################
#a = tf.Variable(np.array([np.exp(1.0)],dtype=np.float32))
#
#with tf.GradientTape() as tape:
#    c = tf.math.log(a)
#
#gradients = tape.gradient(c, a)
#print('log')
#print(c)
#print(gradients)

##################################################
#a = tf.Variable(np.array([3.0],dtype=np.float32))
#
#with tf.GradientTape() as tape:
#    c = tf.math.square(a)
#
#gradients = tape.gradient(c, a)
#print('square')
#print(c)
#print(gradients)

##################################################
#a = tf.Variable(np.array([9.0],dtype=np.float32))
#
#
#with tf.GradientTape() as tape:
#    c = tf.math.sqrt(a)
#
#gradients = tape.gradient(c, a)
#print('sqrt')
#print(c)
#print(gradients)

##################################################
#@tf.function
#def ccc(x):
#    print('a',i)
#    y = tf.math.sqrt(x)
#    return y
#
#a = tf.Variable(np.array([9.0],dtype=np.float32))
#
#print('b')
#
#for i in range(3):
#    #i = tf.constant(i)
#    with tf.GradientTape() as tape:
#        with tf.name_scope('g'):
#            y = ccc(a)
#    gradients = tape.gradient(y, a)


#import timeit
#print(timeit.timeit(lambda: ggg(a), number=10000))

##################################################
#class TestModel(tf.keras.Model):
#    def __init__(self,**kwargs):
#        super(TestModel, self).__init__(**kwargs)
#        self.cnst = tf.constant([1.0])
#        self.v = tf.Variable([1.0])
#        self.layer = tf.keras.layers.Dense(4,input_shape=(3,))
#
#    #@tf.function
#    def call(self,inputs,c):
#        print('call',c)
#        outputs = self.layer(inputs)
#        return outputs
#
#model = TestModel()
##model.compile(loss='mse')
#
#inputs = np.array([[2.0,2.0,2.0],[3.0,3.0,3.0]],dtype=np.float32)
#c = tf.constant(1.)
#print('======start predicts======')
#for i in range(3):
##    model.predict(inputs)
#    print('done')
#
#print('======start grads======')
#for i in range(3):
#    with tf.GradientTape() as tape:
#        outputs = model(inputs,c)
#    gradients = tape.gradient(outputs, model.trainable_variables+[c])
#    #print(gradients)
#    print('done')
#
#print('======variables======')
#for v in model.variables:
#    print(v)
#print('======trainable variables======')
#for v in model.trainable_variables:
#    print(v)
#
#print('======non trainable variables======')
#for v in model.non_trainable_variables:
#    print(v)
#print('======submodules======')
#for v in model.submodules:
#    print(v)
#print('const=',tf.constant(np.array([1.])))    
#print('null=',tf.Variable(None))    
#
l = tf.keras.layers.Dense(3,kernel_initializer='ones')
#a = np.array([[2.],[3.],[3.]],dtype=np.float32)
#b = np.array([[2.],[4.],[3.]],dtype=np.float32)
a = tf.Variable([[2.],[3.],[3.]])
b = tf.Variable([[2.],[4.],[3.]])
with tf.GradientTape() as tape:
    x = a+b
    print(x)
    y = l(x)
    print(y)
print(a)
grads = tape.gradient(y,[a,x])
for v in grads:
    print(v)
    print(v.name)
