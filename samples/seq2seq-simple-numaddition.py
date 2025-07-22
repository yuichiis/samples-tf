#import os
#import pickle
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

# Parameters for the model and dataset.
DIGITS = 3
TRAINING_SIZE = 5000
REVERSE = True
EPOCHS = 30
BATCH_SIZE = 32
UNITS = 128
WORDVECT_SIZE = 16

# Generate the data
class NumAdditionDataset:

    def __init__(
            self,
            corpus_max: int,
            digits: int,
            reverse=True):
        self.vocab_input  = ['0','1','2','3','4','5','6','7','8','9','+',' ']
        self.vocab_target = ['0','1','2','3','4','5','6','7','8','9','+',' ']
        self.dict_input = dict(zip(self.vocab_input,range(len(self.vocab_input))))
        self.dict_target = dict(zip(self.vocab_target,range(len(self.vocab_target))))
        self.corpus_max = corpus_max
        self.digits = digits
        self.input_length = digits*2+1
        self.output_length = digits+1
        self.reverse = reverse

    def dicts(self) -> tuple:
        return (
            self.vocab_input,
            self.vocab_target,
            self.dict_input,
            self.dict_target,
        )

    def generate(self) -> tuple:
        '''generate random sequence'''
        max_num = pow(10,self.digits)
        max_sample = max_num ** 2 - 1
        numbers = np.random.choice(
            max_sample,max_sample,replace=False)
        questions = {}
        size = 0
        for i in range(max_sample):
            num = numbers[i]
            x1,x2 = sorted(divmod(num,max_num))
            question = '%d+%d' % (x1,x2)
            if questions.get(question) is not None:
                continue
            questions[question] = '%d' % (x1+x2)
            size += 1
            if size >= self.corpus_max:
                break
        numbers = None
        sequence = np.zeros([size,self.input_length],dtype=np.int32)
        target = np.zeros([size,self.output_length],dtype=np.int32)
        i = 0
        for question,answer in questions.items():
            question = question + " " * (self.input_length - len(question))
            answer = answer + " " * (self.output_length - len(answer))
            #if self.reverse:
            #    question = question[::-1]
            self.str2seq(
                question,
                self.dict_input,
                sequence[i])
            self.str2seq(
                answer,
                self.dict_target,
                target[i])
            i += 1
        return (sequence,target)

    def str2seq(
        self,
        input_string: str,
        word_dic: dict,
        output_seq: ndarray) -> None:
        '''translate string to sequence'''
        sseq = list(input_string.upper())
        seq_len = len(sseq)
        sp = word_dic[' ']
        bufsz=output_seq.size
        for i in range(bufsz):
            if i<seq_len:
                output_seq[i] = word_dic[sseq[i]]
            else:
                output_seq[i]=sp

    def seq2str(
        self,
        input_seq: ndarray,
        word_dic: dict
        ) -> str:
        '''translate string to sequence'''
        output_str = ''
        bufsz = input_seq.size
        for i in range(bufsz):
            output_str = output_str + word_dic[input_seq[i]]
        return output_str

    #def translate(
    #    self,
    #    model,
    #    input_str: str) -> str:
    #    '''translate sentence'''
    #    inputs = np.zeros([1,self.input_length],dtype=np.int32)
    #    self.str2seq(
    #        input_str,
    #        self.dict_input,
    #        inputs[0])
    #    outputs = model.predict(inputs)
    #    return self.seq2str(
    #        np.argmax(outputs[0],axis=1)),
    #        self.vocab_target)

    def load_data(
        self,
        path: str=None) -> ndarray:
        '''load dataset'''
        #if path is None:
        #    path='numaddition-dataset.pkl'

        #if os.path.exists(path):
        #    with open(path,'rb') as fp:
        #        dataset = pickle.load(fp)
        #else:
        dataset = self.generate()
        #    with open(path,'wb') as fp:
        #        pickle.dump(dataset,fp)
        return dataset


class CustomCallback(keras.callbacks.Callback):
    def __init__(self,**kwargs):
        super(CustomCallback, self).__init__()
        self.gradlog = {}
        self.prev_w = None
    def on_epoch_end(self, epoch, logs=None):
        weights = self.model.get_weights();
        if self.prev_w is None:
            self.prev_w = weights
            return
        num = 0;
        next = []
        for w,prev in zip(weights,self.prev_w):
            name = 'g'+str(num)+'('+str(w.shape)+')'
            if self.gradlog.get(name) is None:
                self.gradlog[name] = [];
            g = prev - w
            self.gradlog[name].append(np.max(np.abs(w)))
            #echo sprintf("%7.2f",strval())." ";
            num += 1
            next.append(w.copy())
        self.prev_w = next


# All the numbers, plus sign and space for padding.
input_length  = DIGITS*2 + 1
output_length = DIGITS + 1

dataset = NumAdditionDataset(TRAINING_SIZE,DIGITS,REVERSE)

print("Generating data...")
questions,answers = dataset.load_data()
corpus_size = len(questions)
print("Total questions:", corpus_size)
input_voc,target_voc,input_dic,target_dic=dataset.dicts()

# Explicitly set apart 10% for validation data that we never train over.
split_at = len(questions) - len(questions) // 10
(x_train, x_val) = questions[:split_at], questions[split_at:]
(y_train, y_val) = answers[:split_at], answers[split_at:]

print("Training Data:")
print(x_train.shape)
print(y_train.shape)

print("Validation Data:")
print(x_val.shape)
print(y_val.shape)

#x_train = keras.utils.to_categorical(
#    x_train.reshape(x_train.size,),
#    num_classes=len(input_voc)
#    ).reshape(x_train.shape[0],x_train.shape[1],len(input_voc))
#x_val = keras.utils.to_categorical(
#    x_val.reshape(x_val.size,),
#    num_classes=len(target_voc)
#    ).reshape(x_val.shape[0],x_val.shape[1],len(target_voc))
#v_x = keras.utils.to_categorical(v_x.reshape(16,), num_classes=10).reshape(4,4,10)

# Build the model

print("Build model...")

model = keras.Sequential([
    layers.Embedding(len(input_dic), WORDVECT_SIZE),
    #keras.Input(shape=(input_length,len(input_dic))),
    # Encoder
    layers.GRU(UNITS,go_backwards=REVERSE),
    # Expand to answer length and peeking hidden states
    layers.RepeatVector(output_length),
    # Decoder
    layers.GRU(UNITS, return_sequences=True),
    # Output
    layers.Dense(
        len(target_dic),
        #activation='softmax',
    ),
])

def sequence_metrics(y_true, y_pred):
    y_pred = tf.reduce_max(y_pred, axis=-1)
    squared_difference = tf.reduce_sum(tf.math.equal(y_true, y_pred), axis=-1)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

model.compile(
    #loss="sparse_categorical_crossentropy",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="adam",
    metrics={"accuracy": sequence_metrics})
model.summary()

# Train the model

#callback = CustomCallback()
history = model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val, y_val),
    #callbacks=[callback],
)

for i in range(10):
    idx = np.random.randint(0,len(questions))
    question = questions[idx]
    input = question.reshape(1,input_length)

    #input = keras.utils.to_categorical(
    #    input.reshape(input.size,),
    #    num_classes=len(input_voc)
    #    ).reshape(input.shape[0],input.shape[1],len(input_voc))

    predict = model.predict(input)
    predict_seq = np.argmax(predict[0].reshape(output_length,len(target_dic)),axis=1)
    predict_str = dataset.seq2str(predict_seq,target_voc)
    question_str = dataset.seq2str(question,input_voc)
    answer_str = dataset.seq2str(answers[idx],target_voc)
    #if dataset.reverse:
    #    question_str = question_str[::-1]
    correct = '*' if predict_str==answer_str else ' '
    print('%s=%s : %s %s' % (question_str,predict_str,correct,answer_str))

plt.figure()
plt.plot(np.array(history.history['accuracy']),label='accuracy')
plt.plot(np.array(history.history['val_accuracy']),label='val_accuracy')
plt.plot(np.array(history.history['loss']),label='loss')
plt.plot(np.array(history.history['val_loss']),label='val_loss')
plt.legend();
plt.title('seq2seq-numaddition')
#wnum = 0
#idx = 0
#fig,axes = plt.subplots(4,5)
#for w in model.get_weights():
#    print(w.shape,np.amax(w))
#    if wnum in [1,2,4,5]:
#        ww = w[:,0:128]
#        axes[idx//5,idx%5].imshow(ww)
#        idx += 1
#        ww = w[:,128:128*2]
#        axes[idx//5,idx%5].imshow(ww)
#        idx += 1
#        ww = w[:,128*2:128*3]
#        axes[idx//5,idx%5].imshow(ww)
#        idx += 1
#    elif wnum in [0,3,6,7]:
#        axes[idx//5,idx%5].imshow(w)
#        idx += 1
#    wnum += 1
#plt.figure()
#for key,gradlog in callback.gradlog.items():
#    plt.plot(np.array(gradlog),label=key)
#plt.legend()
plt.show()
