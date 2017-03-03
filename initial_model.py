from __future__ import print_function
import os
import sys
import numpy as np
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Input, Flatten, RepeatVector, Lambda, Reshape, MaxPooling2D, MaxPooling1D, AveragePooling2D, AveragePooling1D
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, Merge, Permute, Activation
from keras.models import Model, Sequential
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

BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '/20_newsgroup/'
MAX_PASSAGE_LENGTH = 400
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
count = 0
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
    if count > 100000:
        break
    # count += 1
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

with open("reduced-train-preproc-tokenized.json") as train_fp:
# with open("train-preproc-tokenized.json") as train_fp:
    train_instances = json.load(train_fp)

print('Found %s training instances.' % len(train_instances))

train_instances = np.asarray(train_instances)

passages = [[word.lower() for word in passage] for passage in train_instances[:, 0]]
questions = [[word.lower() for word in passage] for passage in train_instances[:, 1]]
labels = train_instances[:, 2]
labels = to_categorical(labels)

p_vocab = set([word for passage in passages for word in passage])
q_vocab = set([word for question in questions for word in question])
vocab = [''] + list(p_vocab | q_vocab)
word_index = {}
for i, word in enumerate(vocab):
    word_index[word] = i

print('Unique vocab entries:', len(word_index))

passages = [[word_index[word] for word in passage] for passage in passages]
questions = [[word_index[word] for word in question] for question in questions]

passages = pad_sequences(passages, maxlen=MAX_PASSAGE_LENGTH, dtype='int32',
                          padding='post', truncating='post', value=0.)
questions = pad_sequences(questions, maxlen=MAX_PASSAGE_LENGTH, dtype='int32',
                          padding='post', truncating='post', value=0.)
labels = pad_sequences(labels, maxlen=MAX_PASSAGE_LENGTH, dtype='int32',
                          padding='post', truncating='post', value=0.)

print('Shape of passage tensor:', passages.shape)
print('Shape of question tensor:', questions.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(passages.shape[0])
np.random.shuffle(indices)
passages = passages[indices]
questions = questions[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * passages.shape[0])

p_train = passages[:-nb_validation_samples]
q_train = questions[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
p_val = passages[-nb_validation_samples:]
q_val = questions[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

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
            embedding_matrix[i] = (np.random.random_sample((EMBEDDING_DIM, )) - .5) * 2

print('Shape of embedding matrix:', embedding_matrix.shape)

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
p_embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_PASSAGE_LENGTH,
                            trainable=False)

q_embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_PASSAGE_LENGTH,
                            trainable=False)

print('Training model.')

passage_net = Sequential()
passage_net.add(p_embedding_layer)
passage_net.add(Activation('tanh'))
passage_net.add(Dropout(0.1))
# passage_net.add(Bidirectional(LSTM(MAX_PASSAGE_LENGTH, return_sequences=True)))
# passage_net.add(GRU(EMBEDDING_DIM, return_sequences=True))
# passage_net.add(LSTM(EMBEDDING_DIM, return_sequences=True))
print("passage layer shape:", passage_net.layers[-1].output_shape)
# passage_net.add(Dropout(0.5))
# question_net.add(Dense(1, activation='sigmoid'))

question_net = Sequential()
question_net.add(q_embedding_layer)
question_net.add(Activation('tanh'))
question_net.add(Dropout(0.1))
# question_net.add(Bidirectional(LSTM(MAX_PASSAGE_LENGTH, return_sequences=True)))
# question_net.add(GRU(EMBEDDING_DIM, return_sequences=True))
# question_net.add(LSTM(EMBEDDING_DIM, return_sequences=True))
print("question layer shape:", question_net.layers[-1].output_shape)
# question_net.add(Dense(1, activation='sigmoid'))

plot(question_net, to_file='question_net.png', show_shapes=True)

merged = Merge([passage_net, question_net], mode='dot')
# merged = Merge([passage_net, question_net], mode='cos')
print("merged layer shape:", question_net.layers[-1].output_shape)


model = Sequential()

model.add(merged)

# model.add(MyLayer(400))
# model.add(Reshape((1, 400, 400)))
# model.add(Permute((0, 2, 1)))
# model.add(MaxPooling2D(pool_size=(1, 25), border_mode='valid'))
# model.add(AveragePooling2D(pool_size=(1, 4), border_mode='valid'))
# model.add(Permute((0, 2, 1)))
model.add(Permute((2, 1)))
model.add(MaxPooling1D(pool_length=25, stride=None, border_mode='valid'))
model.add(AveragePooling1D(pool_length=10, stride=None, border_mode='valid'))
model.add(Permute((2, 1)))
model.add(Flatten())
model.add(Dense(MAX_PASSAGE_LENGTH, activation='tanh')) #significantly improved accuracy
# model.add(Dropout(.2))
model.add(Dense(MAX_PASSAGE_LENGTH, activation='softmax'))


plot(model, to_file='model.png', show_shapes=True)

# train a 1D convnet with global maxpooling
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# happy learning!
# model.fit(x=[passages, questions], y=labels, nb_epoch=2, batch_size=128)
model.fit([p_train, q_train], y_train, validation_data=([p_val, q_val], y_val),
          nb_epoch=100, batch_size=BATCH_SIZE)
# model.fit(p_train, y_train, validation_data=(p_val, y_val),
          # nb_epoch=2, batch_size=128)
