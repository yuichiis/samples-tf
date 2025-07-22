import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

#x = np.array([
#    [-10.0,  1.0, 10.0],
#    [ 10.0,-10.0,  1.0],
#]);
#t = np.array([2,2])
#func = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#x = np.array([
#[-2.0], [2.0] , [0.0]
#]);
#t = np.array([0.0, 1.0 , 0.0])
#func = keras.losses.BinaryCrossentropy(from_logits=True)

def numerical_gradient(
        f,
        *variables,
        h=None):

    if h is None:
        h = 1e-4
    if not callable(f):
        raise Exception("f must callable or array of f and h")

    grads = []
    new_variables = []
    for x in variables:
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        new_variables.append(x)
    variables = new_variables

    for x in variables:
        g = np.zeros_like(x)
        grads.append(g)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        h2 = h*2
        with it:
            while not it.finished:
                idx = it.multi_index
                value = xx[idx]
                x[idx] = value + h
                y1 = f(*variables)
                x[idx] = value - h
                y2 = f(*variables)
                d = (y1 - y2)
                g[idx] = np.sum(d)/h2
                x[idx] = value
                it.iternext()
    return grads


#x = np.array([
#    [0,1,2,9],
#    #[1],
#])
#t = np.array(
#    [1,0]
#)
#x = keras.utils.to_categorical(x.reshape(4,), num_classes=10).reshape(1,4,10)
#x = keras.utils.to_categorical(x.reshape(1,), num_classes=10).reshape(1,1,10)
q = np.array([
    [[1,0,0],[0,1,0]],
    [[1,0,0],[0,1,0]],
],dtype=np.float32)
v = np.array([
    [[1,0,0],[0,1,0],[0,0,1],[0,0,0]],
    [[1,0,0],[0,1,0],[0,0,1],[0,0,0]],
],dtype=np.float32)

def my_init(shape, dtype=None):
    return tf.fill(shape,2.)
#func = keras.layers.GRU(
#        10,# units
#        input_shape=(4,10),
#        #input_shape=(1,10),
#        activation=None,
#        #recurrent_activation=None,
#        return_sequences=True,
#        kernel_initializer='ones',
#        recurrent_initializer='ones',
#        bias_initializer='zeros',
#        #reset_after=False,
#    )
#kernel = np.full([10,3],0.01);
#recurrent = np.full([3,3],0.01);
#bias = np.zeros([3]);
#func.build((4,10))
#func.weights[0][...] = kernel[...]
#func = keras.layers.Attention(return_attention_scores=True)
func = keras.layers.Attention()
#func = tf.matmul
def dsoftmax(doutput,output):
    dx = output * doutput
    sumdx = tf.reduce_sum(dx, axis=-1, keepdims=True)
    dx -= output * sumdx
    return dx

#xx = tf.Variable(x)
#tt = tf.Variable(t)
qq = tf.Variable(q)
vv = tf.Variable(v)
with tf.GradientTape() as g:
    #yy = func(tt,xx)
    yy = func([qq,vv])
    #yy = func(qq,vv,transpose_b=True)
print('========vector===========')
print('yy:',yy)
print('========dQuery,dValue===========')
dy = g.gradient(yy,[qq,vv])
print('dy:',dy)
#dy_qq = g.gradient(yy,qq)
#dy_vv = g.gradient(yy,vv)
#print('dy:',dy_qq)
#print('dy:',dy_vv)


## simulation forwards
print('========forward===========')
scores = tf.matmul(qq,vv,transpose_b=True)
print('scores:',scores)
weights = tf.nn.softmax(scores)
print('weights:',weights)
vector = tf.matmul(weights,vv)
print('vector:',vector)

print('========backward===========')
## simulation backwards
dyy = tf.Variable(np.ones([2,2,3],dtype=np.float32))
# forward: vector = matmul(weights,vv)
print('dWeights=matmul(dyy,vv,transpose_b=True)')
dWeights=tf.matmul(dyy,vv,transpose_b=True)
print(dWeights)
print('dValue=matmul(weights,dyy,transpose_a=True)')
dValue=tf.matmul(weights,dyy,transpose_a=True)
print(dValue)
# forward: weights = softmax(scores)
dScores = dsoftmax(dWeights,weights)
print('dscores = dsoftmax(dWeights,weights)')
print(dScores)
# forward: scores = matmul(query,value)
print('dQuery=matmul(dScores,vv)')
dQuery=tf.matmul(dScores,vv)
print(dQuery)
print('dKey=matmul(dScores,query,transpose_a=True)')
dKey=tf.matmul(dScores,qq,transpose_a=True)
print(dKey)
dValue += dKey
print('dValue += dKey')
print(dValue)
#grads = numerical_gradient(func,x,h=1e-3)
#print(grads[0])
