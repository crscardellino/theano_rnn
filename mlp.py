# -*- coding: utf-8 -*-

import numpy as np
import re
import sys
import theano
import theano.tensor as T

from nltk import corpus
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text

np.random.seed(123)  # For reproducibility


def process_newsgroups_document(document):
    # To simplify, we ignore everything thas isn't a word
    document = re.sub(r"[^a-zA-Z]", " ", document)

    # We only make use of lower case words
    words = document.lower().split()

    # We filter out every stopword for the english language
    stopwords = set(corpus.stopwords.words("english"))
    document = " ".join([word for word in words if word not in stopwords])

    return document

print "Fetching and processing 20 Newsgroup"
sys.stdout.flush()
newsgroups = fetch_20newsgroups(subset='all')
vectorizer = text.CountVectorizer(analyzer='word', preprocessor=process_newsgroups_document, max_features=5000)
newsgroups_dataset = vectorizer.fit_transform(newsgroups.data).todense().astype(theano.config.floatX)
newsgroups_target = newsgroups.target
ng_X_train, ng_X_test, ng_y_train, ng_y_test = train_test_split(newsgroups_dataset, newsgroups_target, test_size=0.2)
ng_X_train = ng_X_train[:4000]
ng_y_train = ng_y_train[:4000]
ng_X_test = ng_X_test[4000:5000]
ng_y_test = ng_y_test[4000:5000]

print "Converting train variables to theano"
sys.stdout.flush()
# Convert the data to theano shared variables
ng_X_train = theano.shared(ng_X_train, borrow=True)
ng_y_train = theano.shared(ng_y_train, borrow=True)

print "Setting up parameters"
sys.stdout.flush()
N = newsgroups_dataset.shape[0]  # Number of examples in the dataset.
n_input = newsgroups_dataset.shape[1]  # Number of features of the dataset. Input of the Neural Network.
n_output = len(newsgroups.target_names)  # Number of classes in the dataset. Output of the Neural Network.
n_h1 = 2500  # Size of the first layer
n_h2 = 1000  # Size of the second layer
alpha = 0.01  # Learning rate parameter
lambda_reg = 0.01  # Lambda value for regularization
epochs = 1000  # Number of epochs for gradient descent
batch_size = 128  # Size of the minibatches to perform sgd
train_batches = ng_X_train.get_value().shape[0] / batch_size

print "Defining computational graph"
sys.stdout.flush()
# Stateless variables to handle the input
index = T.lscalar('index')  # Index of a minibatch
X = T.matrix('X')
y = T.lvector('y')

# First layer weight matrix and bias
W1 = theano.shared(
    value=np.random.uniform(
        low=-np.sqrt(6. / (n_input + n_h1)),
        high=np.sqrt(6. / (n_input + n_h1)),
        size=(n_input, n_h1)
    ).astype(theano.config.floatX),
    name='W1',
    borrow=True
)
b1 = theano.shared(
    value=np.zeros((n_h1,), dtype=theano.config.floatX),
    name='b1',
    borrow=True
)

# Second layer weight matrix and bias
W2 = theano.shared(
    value=np.random.uniform(
        low=-np.sqrt(6. / (n_h1 + n_h2)),
        high=np.sqrt(6. / (n_h1 + n_h2)),
        size=(n_h1, n_h2)
    ).astype(theano.config.floatX),
    name='W2',
    borrow=True
)
b2 = theano.shared(
    value=np.zeros((n_h2,), dtype=theano.config.floatX),
    name='b2',
    borrow=True
)

# Output layer weight matrix and bias
W3 = theano.shared(
    value=np.random.uniform(
        low=-np.sqrt(6. / (n_h2 + n_output)),
        high=np.sqrt(6. / (n_h2 + n_output)),
        size=(n_h2, n_output)
    ).astype(theano.config.floatX),
    name='W3',
    borrow=True
)
b3 = theano.shared(
    value=np.zeros((n_output,), dtype=theano.config.floatX),
    name='b3',
    borrow=True
)

z1 = T.dot(X, W1) + b1  # Size: N x n_h1
a1 = T.tanh(z1)  # Size: N x n_h1

z2 = T.dot(a1, W2) + b2  # Size: N x n_h2
a2 = T.tanh(z2)  # Size: N x n_h2

z3 = T.dot(a2, W3) + b3  # Size: N x n_output
y_out = T.nnet.softmax(z3)  # Size: N x n_output

y_pred = T.argmax(y_out, axis=1)  # Size: N

# Regularization term to sum in the loss function
loss_reg = 1./N * lambda_reg/2 * (T.sum(T.sqr(W1)) + T.sum(T.sqr(W2)) + T.sum(T.sqr(W3)))

# Loss function
loss = T.nnet.categorical_crossentropy(y_out, y).mean() + loss_reg

print "Compiling theano functions"
sys.stdout.flush()
# Define the functions
forward_propagation = theano.function([X], y_out)
loss_calculation = theano.function([X, y], loss)
predict = theano.function([X], y_pred)

print "Getting gradients"
sys.stdout.flush()
dJdW1 = T.grad(loss, wrt=W1)
dJdb1 = T.grad(loss, wrt=b1)
dJdW2 = T.grad(loss, wrt=W2)
dJdb2 = T.grad(loss, wrt=b2)
dJdW3 = T.grad(loss, wrt=W3)
dJdb3 = T.grad(loss, wrt=b3)

print "Setting updates"
sys.stdout.flush()
updates = [
    (W1, W1 - alpha * dJdW1),  # Update step. W1 = W1 - alpha * dJdW1
    (b1, b1 - alpha * dJdb1),  # Update step. b1 = b1 - alpha * dJdb1
    (W2, W2 - alpha * dJdW2),  # Update step. W2 = W2 - alpha * dJdW2
    (b2, b2 - alpha * dJdb2),  # Update step. b2 = b2 - alpha * dJdb2
    (W3, W3 - alpha * dJdW3),  # Update step. W3 = W3 - alpha * dJdW3
    (b3, b3 - alpha * dJdb3),  # Update step. b3 = b3 - alpha * dJdb3
]

print "Compiling gradient step"
sys.stdout.flush()
gradient_step = theano.function(
    inputs=[index],
    outputs=loss,
    updates=updates,
    givens={
        X: ng_X_train[index * batch_size: (index + 1) * batch_size],
        y: ng_y_train[index * batch_size: (index + 1) * batch_size]
    }
)

print "Starting training"
sys.stdout.flush()
for i in xrange(epochs):  # We train for epochs times
    for mini_batch in xrange(train_batches):
        gradient_step(mini_batch)

    if i % 50 == 0:
        print "Loss for iteration {}: {}".format(
            i, loss_calculation(ng_X_train.get_value(), ng_y_train.get_value())
        )
        sys.stdout.flush()

print "Training finished. Getting some results."
sys.stdout.flush()

predictions = predict(ng_X_test)

print "Accuracy: {:.3f}".format(accuracy_score(ng_y_test, predictions))
sys.stdout.flush()

print "Classification report"
print classification_report(ng_y_test, predictions, labels=newsgroups.target_names)
sys.stdout.flush()
