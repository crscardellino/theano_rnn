# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import theano
import theano.tensor as T

class Corpus(object):
    def __init__(self, corpus_path):
        self.corpus = {}
        self.length = 0

        for fname in os.listdir(corpus_path):
            fpath = os.path.join(corpus_path, fname)

            with open(fpath, "r") as f:
                self.corpus[fname.replace(".txt", "")] = f.read().decode("utf-8")

        characters = set()

        for work_name, work in self.corpus.iteritems():
            for c in work:
                characters.add(c)
                self.length += 1

        self.characters = sorted(characters)

    def character_encoder(self, char):
        vector = np.zeros((len(self.characters),), dtype='int64')
        vector[self.characters.index(char)] = 1

        return vector

    def __iter__(self):
        for work_name, work in self.corpus.iteritems():
            for char in work:
                yield self.character_encoder(char)
            yield self.character_encoder(u"\n")

    def __len__(self):
        return self.length

corpus = Corpus("corpus/borges")

NT = len(corpus)  # Number of examples (timesteps)
n_in = len(corpus.characters)  # Size of the input data (one-hot vector of a character)
n_out = len(corpus.characters)  # Size of the output data (one-hot vector of a character)
n_h = 50  # Size of the hidden layer

# Stateless variables to handle the input
X = T.matrix('X')
y = T.lvector('y')

W_hx = theano.shared(
    value=np.random.uniform(
        low=-1.0,
        high=1.0,
        size=(n_in, n_h)
    ).astype(theano.config.floatX),
    name='W_hx',
    borrow=True
)

b_h = theano.shared(
    value=np.zeros(n_h, dtype=theano.config.floatX),
    name='b_h',
    borrow=True
)

W_hh = theano.shared(
    value=np.random.uniform(
        low=-1.0,
        high=1.0,
        size=(n_h, n_h)
    ).astype(theano.config.floatX),
    name='W_hh',
    borrow=True
)

W_S = theano.shared(
    value=np.random.uniform(
        low=-1.0,
        high=1.0,
        size=(n_h, n_out)
    ).astype(theano.config.floatX),
    name='W_S',
    borrow=True
)

b_S = theano.shared(
    value=np.zeros(n_out, dtype=theano.config.floatX),
    name='b_S',
    borrow=True
)

h0 = theano.shared(
    value=np.zeros(n_h, dtype=theano.config.floatX),
    name='h0',
    borrow=True
)


def forward_propagation_step(x_t, h_t_prev, W_hx, W_hh, b_h, W_S, b_S):
    h_t = T.tanh(T.dot(x_t, W_hx) + T.dot(h_t_prev, W_hh) + b_h)
    y_t = T.nnet.softmax(T.dot(h_t, W_S) + b_S)

    return [h_t, y_t]

[h, y_out], _ = theano.scan(
    forward_propagation_step,
    sequences=X,
    outputs_info=[h0, None],
    non_sequences=[W_hx, W_hh, b_h, W_S, b_S],
    truncate_gradient=100,
    n_steps=X.shape[0]
)

p_y_given_x = y_out[:, 0, :]

y_pred = T.argmax(p_y_given_x, axis=1)

loss = T.nnet.categorical_crossentropy(p_y_given_x, y).mean()

dWhx = T.grad(loss, wrt=W_hx)
dWhh = T.grad(loss, wrt=W_hh)
dbh = T.grad(loss, wrt=b_h)
dWS = T.grad(loss, wrt=W_S)
dbS = T.grad(loss, wrt=b_S)

forward_propagation = theano.function([X], y_out)
loss_calculation = theano.function([X, y], loss)
predict = theano.function([X], y_pred)
# bbtt = theano.function([X, y], [dWhx, dWhh, dbh, dWS, dbS])

alpha = T.scalar('alpha')

updates = [
    (W_hx, W_hx - alpha * dWhx),
    (W_hh, W_hh - alpha * dWhh),
    (b_h, b_h - alpha * dbh),
    (W_S, W_S - alpha * dWS),
    (b_S, b_S - alpha * dbS)
]

gradient_step = theano.function(
    inputs=[X, y, alpha],
    outputs=loss,
    updates=updates
)

X_train = []
y_train = []

for char in corpus:
    X_train.append(char)
    y_train.append(np.where(char == 1)[0][0])

X_train = np.vstack(X_train[:-1])
y_train = np.array(y_train[1:])

for i in xrange(1, 1000):  # We train for epochs times
    for j in xrange(y_train.shape[0], 10):
        gradient_step(X_train[j:j+10], y_train[j:j+10], 0.001)

    if i % 50 == 0:
        print >> sys.stderr, "Loss for iteration {}: {}".format(
            i, loss_calculation(X_train, y_train)
        )

        # Generate a 1000 characters text

        random_char = corpus.characters[np.random.randint(28, 82)]

        characters = [(
                random_char,
                corpus.character_encoder(random_char)
            )]
        # The first character is alphabetic random

        for j in xrange(1000):
            char_vectors = np.vstack([vector for _, vector in characters])
            next_char = corpus.characters[predict(char_vectors)[-1]]
            characters.append((
                    next_char,
                    corpus.character_encoder(next_char)
                ))

        print "".join([char for char, _ in characters])
