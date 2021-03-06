#!/usr/bin/env python
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
        return self.characters.index(char)

    def __iter__(self):
        for work_name, work in self.corpus.iteritems():
            for char in work:
                yield self.character_encoder(char)
            yield self.character_encoder(u"\n")

    def __len__(self):
        return self.length

print >> sys.stderr, "Getting corpus"
sys.stderr.flush()
corpus = Corpus("corpus/borges")

NT = len(corpus)  # Number of examples (timesteps)
n_in = len(corpus.characters)  # Size of the input data (one-hot vector of a character)
n_out = len(corpus.characters)  # Size of the output data (one-hot vector of a character)
n_h = 50  # Size of the hidden layer

print >> sys.stderr, "Declaring Theano variables"
sys.stderr.flush()
# Stateless variables to handle the input
X = T.ivector('X')
y = T.ivector('y')

W_hx = theano.shared(
    value=np.random.uniform(
        low=-1.0,
        high=1.0,
        size=(n_h, n_in)
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
        size=(n_out, n_h)
    ).astype(theano.config.floatX),
    name='W_S',
    borrow=True
)

b_S = theano.shared(
    value=np.zeros(n_out, dtype=theano.config.floatX),
    name='b_S',
    borrow=True
)


def forward_propagation_step(x_t, h_t_prev, W_hx, W_hh, b_h, W_S, b_S):
    h_t = T.tanh(W_hx[:, x_t] + T.dot(h_t_prev, W_hh) + b_h)
    y_t = T.nnet.softmax(T.dot(W_S, h_t) + b_S)

    return [h_t, y_t]

print >> sys.stderr, "Declaring Theano Loop and Graph"
sys.stderr.flush()

[h, y_out], _ = theano.scan(
    forward_propagation_step,
    sequences=X,
    outputs_info=[dict(initial=T.zeros(n_h)), None],
    non_sequences=[W_hx, W_hh, b_h, W_S, b_S],
    truncate_gradient=50,
    n_steps=X.shape[0]
)

p_y_given_x = y_out[:, 0, :]

y_pred = T.argmax(p_y_given_x, axis=1)

loss = T.nnet.categorical_crossentropy(p_y_given_x, y).mean()

print >> sys.stderr, "Getting gradients"
sys.stderr.flush()
dWhx = T.grad(loss, wrt=W_hx)
dWhh = T.grad(loss, wrt=W_hh)
dbh = T.grad(loss, wrt=b_h)
dWS = T.grad(loss, wrt=W_S)
dbS = T.grad(loss, wrt=b_S)

print >> sys.stderr, "Compiling functions"
forward_propagation = theano.function([X], y_out)
loss_calculation = theano.function([X, y], loss)
predict = theano.function([X], y_pred)
# bbtt = theano.function([X, y], [dWhx, dWhh, dbh, dWS, dbS])

alpha = T.scalar('alpha')

print >> sys.stderr, "Declaring gradients"
sys.stderr.flush()
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


print >> sys.stderr, "Getting dataset"
sys.stderr.flush()

X_train = []
y_train = []

for char in corpus:
    X_train.append(char)
    y_train.append(char)

X_train = np.array(X_train[:-1]).astype('int32')
y_train = np.array(y_train[1:]).astype('int32')

print >> sys.stderr, "Begin training"
sys.stderr.flush()

for i in xrange(1, 1000):  # We train for epochs times
    for j in xrange(y_train.shape[0], 128):
        gradient_step(X_train[j:j+128], y_train[j:j+128], 0.01)

    if i % 100 == 0:
        print >> sys.stderr, "Loss for iteration {}: {}".format(
            i, loss_calculation(X_train, y_train)
        )
        sys.stderr.flush()

        # Generate a 2000 characters text

        characters_indexes = [np.random.randint(28, 82)]
        # The first character is alphabetic random

        for j in xrange(2000):
            characters_indexes.append(predict(np.hstack(characters_indexes).astype('int32'))[-1])

        print "".join([corpus.characters[char] for char in characters_indexes])
        print
        print "#" * 100
        print
        sys.stdout.flush()
