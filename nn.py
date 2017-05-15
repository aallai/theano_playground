import theano
import numpy

from theano import tensor as T

class ReluConv:

    # Assumes square input.
    def __init__(self, input, input_size, filter_size, stride, input_channels, num_filters, relu_leak = 0):


        # TODO: look up how to initialze weights.
        self.W = theano.shared(value = numpy.random.randn(num_filters, input_channels, filter_size, filter_size), borrow = True)
        self.b = theano.shared(value =numpy.zeros((num_filters,)), borrow = True)

        self.convnet = T.nnet.conv2d(input, self.W, border_mode = 'valid', subsample = (stride, stride))

        self.output = T.nnet.relu(self.convnet + self.b.dimshuffle('x', 0, 'x', 'x'), relu_leak)
        self.flat_output = self.output.flatten(2)
        self.params = [self.W, self.b]

class ReluFC:

    def __init__(self, input, input_dim, output_dim, linear = False, relu_leak = 0):

        # W and b are actually tranposed here to accomodate the row-vector format of the input.
        self.W = theano.shared(value = numpy.random.randn(input_dim, output_dim), name = 'W')
        self.b = theano.shared(value = numpy.random.randn(output_dim), name = 'b')

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
MINIBATCH_SIZE = 1000

import gzip, cPickle

if __name__ == '__main__':

    f = gzip.open(r'C:\Users\alexa\DeepLearningTutorials\data\mnist.pkl.gz')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    x = T.dtensor4('x')
    y = T.dvector('y')
    index = T.iscalar('index')

    conv_layer = ReluConv(input = x, input_size = IMAGE_SIZE, filter_size = FILTER_SIZE,
        stride = STRIDE, input_channels = 1, num_filters = NUM_FILTERS)

    conv_layer_output_dim = (((IMAGE_SIZE - FILTER_SIZE) / STRIDE + 1) ** 2) * NUM_FILTERS

    output_layer = ReluFC(conv_layer.flat_output, conv_layer_output_dim, 1, True)

    cost = output_layer.loss(y)

    params = conv_layer.params + output_layer.params

    grads = T.grad(cost, params)

    learning_rate = 0.15

    updates = [(param, param - learning_rate * grad) for param, grad in zip(params, grads)]

    train_model = theano.function( [index], cost, updates=updates,
        givens={
            x: train_set_x[index * MINIBATCH_SIZE: (index + 1) * MINIBATCH_SIZE].reshape((MINIBATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE)),
            y: train_set_y[index * MINIBATCH_SIZE: (index + 1) * MINIBATCH_SIZE]
        }
    )

    for i in range(len(train_set[0]) / MINIBATCH_SIZE):
        c = train_model(i)

        print "Cost at iteration {}: {}".format(i, c)

    predict = theano.function([], output_layer.output, givens = {x: test_set_x.reshape((10000, 1, IMAGE_SIZE, IMAGE_SIZE))})

    labels = predict()

    for i in range(50):
        print "Predicted: {}, actual: {}".format(labels[i], test_set_y.get_value()[i])

