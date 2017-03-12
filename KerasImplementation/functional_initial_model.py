from __future__ import print_function
import os
import sys
import numpy as np
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, merge
from keras.models import Model, Sequential
from keras import backend as K
import sys
import json

BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '/20_newsgroup/'
MAX_PASSAGE_LENGTH = 400
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1

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
    if count > 10000:
        break
    count += 1
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

with open("reduced-train-preproc-tokenized.json") as train_fp:
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
embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_PASSAGE_LENGTH,
                            trainable=False)

print('Training model.')

p_sequence_input = Input(shape=(MAX_PASSAGE_LENGTH,), dtype='int32')
p_embedded_sequences = embedding_layer(p_sequence_input)
passage_net = Bidirectional(LSTM(MAX_PASSAGE_LENGTH, unroll=True))(p_embedded_sequences)
print("passage layer shape:", K.shape(passage_net))
# passage_net.add(Dropout(0.5))
# question_net.add(Dense(1, activation='sigmoid'))


q_sequence_input = Input(shape=(MAX_PASSAGE_LENGTH,), dtype='int32')
q_embedded_sequences = embedding_layer(p_sequence_input)
question_net = Bidirectional(LSTM(MAX_PASSAGE_LENGTH, unroll=True))(q_embedded_sequences)
print("question layer shape:", K.shape(question_net))
# question_net.add(Dropout(0.5))
# question_net.add(Dense(1, activation='sigmoid'))

merged = merge([passage_net, question_net], mode='dot')
print("Dotted layer shape:", K.shape(merged))
merged = K.sum(merged, axis=1)
print("Merged layer shape:", K.shape(merged))

main_output = Dense(MAX_PASSAGE_LENGTH, activation='softmax')(merged)

# train a 1D convnet with global maxpooling
model = Model([p_sequence_input, q_sequence_input], main_output)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# happy learning!
# model.fit(x=[passages, questions], y=labels, nb_epoch=2, batch_size=32)
model.fit([p_train, q_train], y_train, validation_data=([p_val, q_val], y_val),
          nb_epoch=2, batch_size=32)
# model.fit(p_train, y_train, validation_data=(p_val, y_val),
          # nb_epoch=2, batch_size=128)
