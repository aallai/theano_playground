import theano
import numpy
from math import sqrt

from theano import tensor as T

def relu_weights(input_dim, *args):
    return numpy.random.randn(*args) * sqrt(2.0 / input_dim)

class ReluConv:

    # Assumes square input.
    def __init__(self, input, input_size, filter_size, stride, input_channels, num_filters, relu_leak = 0):


        # TODO: look up how to initialze weights.
        self.W = theano.shared(value = relu_weights(filter_size**2, num_filters, input_channels, filter_size, filter_size), borrow = True)
        self.b = theano.shared(value = numpy.zeros((num_filters,)), borrow = True)

        self.convnet = T.nnet.conv2d(input, self.W, border_mode = 'valid', subsample = (stride, stride))

        self.output = T.nnet.relu(self.convnet + self.b.dimshuffle('x', 0, 'x', 'x'), relu_leak)

        self.flat_output = self.output.flatten(2)

        self.params = [self.W, self.b]

class ReluFC:

    def __init__(self, input, input_dim, output_dim, linear = False, relu_leak = 0):

        # W and b are actually tranposed here to accomodate the row-vector format of the input.
        self.W = theano.shared(value = relu_weights(input_dim, input_dim, output_dim), name = 'W')
        self.b = theano.shared(value = numpy.zeros((output_dim,)), name = 'b')

        activation = T.dot(input, self.W) + self.b

        if linear:
            self.output = activation
        else:
            self.output = T.nnet.relu(activation, relu_leak)

        self.params = [self.W, self.b]

    def loss(self, y):
        return T.mean((self.output - y.dimshuffle(0, 'x')) ** 2)

def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, shared_y

IMAGE_SIZE = 28
FILTER_SIZE = 4
STRIDE = 1
NUM_FILTERS = 16
NUM_LABELS = 10
MINIBATCH_SIZE = 100

import gzip, cPickle

if __name__ == '__main__':

    theano.config.floatX = 'float32'

    f = gzip.open(r'C:\Users\alexa\DeepLearningTutorials\data\mnist.pkl.gz')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    x = T.fmatrix('x')
    y = T.fvector('y')
    index = T.iscalar('index')

    #hidden_layer = ReluConv(input = x, input_size = IMAGE_SIZE, filter_size = FILTER_SIZE,
    #    stride = STRIDE, input_channels = 1, num_filters = NUM_FILTERS)

    #hidden_layer_output_dim = (((IMAGE_SIZE - FILTER_SIZE) / STRIDE + 1) ** 2) * NUM_FILTERS

    # Layer of size 2n + d
    HIDDEN_LAYER_SIZE = len(train_set[0]) * 2 + IMAGE_SIZE**2

    hidden_layer = ReluFC(x, IMAGE_SIZE**2, HIDDEN_LAYER_SIZE)
    output_layer = ReluFC(hidden_layer.output, HIDDEN_LAYER_SIZE, 1, True)

    cost = output_layer.loss(y)

    params = hidden_layer.params + output_layer.params
    velocities = [theano.shared(value = numpy.zeros(param.get_value().shape)) for param in params]
    grads = T.grad(cost, params)

    momentum = 0.5
    lr = 0.01

    #updates = [(param, param - lr* grad) for param, grad in zip(params, grads)]
    updates = [(velocity, momentum * velocity + lr * grad) for velocity, grad in zip(velocities, grads)]
    updates += [(param, param - velocity) for param, velocity in zip(params, velocities)]

    train_model = theano.function([index], cost, updates=updates,
        givens = {
            #x: train_set_x[index * MINIBATCH_SIZE: (index + 1) * MINIBATCH_SIZE].reshape((MINIBATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE)),
            x: train_set_x[index * MINIBATCH_SIZE: (index + 1) * MINIBATCH_SIZE],
            y: train_set_y[index * MINIBATCH_SIZE: (index + 1) * MINIBATCH_SIZE]
        }
    )

    epochs = 10

    for e in range(epochs):
        for i in range(len(train_set[0]) / MINIBATCH_SIZE):

            c = train_model(i)
            print "Average squared loss at iteration {}: {}".format(e * MINIBATCH_SIZE + i, c)

    print "Total average squared loss: {}".format(total_cost())

    #predict = theano.function([], output_layer.output, givens = {x: test_set_x.reshape((10000, 1, IMAGE_SIZE, IMAGE_SIZE))})
    predict = theano.function([], output_layer.output, givens = { x: train_set_x[:100] })

    labels = predict()

    for i in range(100):
        print "Predicted: {}, actual: {}".format(labels[i], train_set_y.get_value()[i])

