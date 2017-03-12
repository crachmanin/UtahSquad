from __future__ import print_function
import os
import sys
import numpy as np
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Input, Flatten, RepeatVector, Lambda, Reshape, MaxPooling2D, MaxPooling1D, AveragePooling2D, AveragePooling1D, Highway
from keras.layers import TimeDistributedDense, Dense, Dropout, Embedding, LSTM, GRU, SimpleRNN, Bidirectional, Merge, Permute, Activation, BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import backend as K
from keras.engine.topology import Layer
from keras.utils.visualize_util import plot
import sys
import json

class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            # Create a trainable weight variable for this layer.
            super(MyLayer, self).build()  # Be sure to call this somewhere!

        def call(self, x, mask=None):
            return K.sum(x, axis=2)

BASE_DIR = '..'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '/data'
MAX_PASSAGE_LENGTH = 100
MAX_QUESTION_LENGTH = 30
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

print('Processing text dataset')

fn = os.path.join(TEXT_DATA_DIR, "sent-train-preproc.json")
with open(fn) as train_fp:
    train_instances = json.load(train_fp)

train_instances.sort(key=lambda x: len(x[0]))

print('Found %s training instances.' % len(train_instances))

train_instances = np.asarray(train_instances)

nb_validation_samples = int(VALIDATION_SPLIT * train_instances.shape[0])
validation_rows = np.random.choice(len(train_instances), nb_validation_samples)
validation_data = train_instances[validation_rows]
validation_rows = set(validation_rows)
train_rows = [x for x in range(len(train_instances)) if x not in validation_rows]
train_data = train_instances[train_rows]

train_passages = [[word.lower() for word in passage] for passage in train_data[:, 0]]
train_questions = [[word.lower() for word in passage] for passage in train_data[:, 1]]
train_labels = train_data[:, 2]
# train_labels = train_data[:, 3]
train_labels = to_categorical(train_labels)

validation_passages = [[word.lower() for word in passage] for passage in validation_data[:, 0]]
validation_questions = [[word.lower() for word in passage] for passage in validation_data[:, 1]]
validation_labels = validation_data[:, 2]
# validation_labels = validation_data[:, 3]
validation_labels = to_categorical(validation_labels)

#add validation vocab in final model
vocab_fn = os.path.join(TEXT_DATA_DIR, "vocab.json")
with open(vocab_fn) as fp:
    vocab = json.load(fp)
vocab = ['!OOV!'] + vocab

word_index = {}
for i, word in enumerate(vocab):
    word_index[word] = i

print('Unique vocab entries:', len(word_index))

train_passages = [[word_index[word] for word in passage] for passage in train_passages]
train_questions = [[word_index[word] for word in question] for question in train_questions]

#set word to zero if out of vocab
validation_passages = [[word_index[word] if word in word_index else 0 for word in passage]
                       for passage in validation_passages]
validation_questions = [[word_index[word] if word in word_index else 0 for word in question]
                        for question in validation_questions]

p_train = pad_sequences(train_passages, maxlen=MAX_PASSAGE_LENGTH, dtype='int32',
                          padding='post', truncating='post', value=0.)
q_train = pad_sequences(train_questions, maxlen=MAX_QUESTION_LENGTH, dtype='int32',
                          padding='post', truncating='post', value=0.)
y_train = pad_sequences(train_labels, maxlen=MAX_PASSAGE_LENGTH, dtype='int32',
                          padding='post', truncating='post', value=0.)

p_val = pad_sequences(validation_passages, maxlen=MAX_PASSAGE_LENGTH, dtype='int32',
                          padding='post', truncating='post', value=0.)
q_val = pad_sequences(validation_questions, maxlen=MAX_QUESTION_LENGTH, dtype='int32',
                          padding='post', truncating='post', value=0.)
y_val = pad_sequences(validation_labels, maxlen=MAX_PASSAGE_LENGTH, dtype='int32',
                          padding='post', truncating='post', value=0.)

print('Preparing embedding matrix.')

# prepare embedding matrix
nb_words = len(word_index)
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be random vectors between -1 and 1
        embedding_matrix[i] = embedding_vector
    else:
        # zero for non words at end of passage
        if word != '':
            embedding_matrix[i] = (np.random.random_sample((EMBEDDING_DIM, )) - .5) * .1

print('Shape of embedding matrix:', embedding_matrix.shape)

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
p_embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_PASSAGE_LENGTH,
                            mask_zero=True,
                            trainable=False)

q_embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_QUESTION_LENGTH,
                            mask_zero=True,
                            trainable=False)

print('Training model.')

#try GRUS
passage_net = Sequential()
passage_net.add(p_embedding_layer)
passage_net.add(Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True)))
passage_net.add(Dropout(0.2))
passage_net.add(Activation('tanh'))
print("passage layer shape:", passage_net.layers[-1].output_shape)
# passage_net.add(Dropout(0.5))

question_net = Sequential()
question_net.add(q_embedding_layer)
question_net.add(Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True)))
question_net.add(Dropout(0.2))
question_net.add(Activation('tanh'))
print("question layer shape:", question_net.layers[-1].output_shape)
# question_net.add(Dense(1, activation='sigmoid'))

plot(question_net, to_file='question_net.png', show_shapes=True)

c_to_q = Merge([passage_net, question_net], mode='dot')

cont_to_query = Sequential()
cont_to_query.add(c_to_q)
cont_to_query.add(Dense(1, activation='softmax'))
cont_to_query.add(Permute((2, 1)))
cont_to_query.add(Flatten())
cont_to_query.add(RepeatVector(2 * EMBEDDING_DIM))
cont_to_query.add(Permute((2, 1)))
cont_to_query.add(Dropout(0.2))

merged = Merge([passage_net, cont_to_query], mode='mul')

model = Sequential()
model.add(merged)
# model.add(Bidirectional(LSTM(EMBEDDING_DIM, return_sequences=True)))
# model.add(Permute((2, 1)))
# model.add(AveragePooling1D(pool_length=EMBEDDING_DIM * 2, stride=None, border_mode='valid'))
# model.add(Permute((2, 1)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(MAX_PASSAGE_LENGTH, activation='softmax'))


plot(model, to_file='model.png', show_shapes=True)

# train a 1D convnet with global maxpooling
# adam = Adam(lr=.0001, clipnorm=10)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc', 'recall'])
              # metrics=['recall'])

# happy learning!
# model.fit(x=[passages, questions], y=labels, nb_epoch=2, batch_size=128)
model.fit([p_train, q_train], y_train, validation_data=([p_val, q_val], y_val),
          nb_epoch=10, batch_size=BATCH_SIZE)
probs = model.predict([p_val, q_val])
preds = np.argmax(probs, axis=1)
correct = 0
for i in xrange(len(preds)):
    pred = preds[i]
    pred_str = vocab[p_val[i][pred]]
    gt = np.argmax(y_val[i])
    gt_str = vocab[p_val[i][gt]]
    print("Prediction: {}, Ground Truth: {}".format(pred, gt))
    print("Prediction: {}, Ground Truth: {}".format(pred_str.encode('utf-8'), gt_str.encode('utf-8')))
    print()
print(correct)
print(max(preds))
# model.fit(p_train, y_train, validation_data=(p_val, y_val),
          # nb_epoch=2, batch_size=128)
