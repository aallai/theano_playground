import theano
import numpy

from theano import tensor as T

class ReluConv:

    # Assumes square input.
    def __init__(self, input, input_size, filter_size, stride, input_channels, output_channels, relu_leak):


        # TODO: look up how to initialze weights.
        self.filters = theano.shared(value = [[numpy.random.randn(filter_size, filter_size) for i in range(input_channels)] for i in range(output_channels)], borrow = True)

        self.convnet = T.nnet.conv2d(input, self.filters, border_mode = 'valid', subsample = (stride, stride))

        self.output = T.nnet.relu(self.convnet, relu_leak)

class ReluFC:

    def __init__(self, input, input_dim, output_dim, linear = False, relu_leak = 0):

        # W and b are actually tranposed here to accomodate the row-vector format of the input.
        self.W = theano.shared(value = numpy.random.randn(input_dim, ouput_dim), name = 'W')
        self.b = theano.shared(value = numpy.random.randn(output_dim), name = 'b')

        activation = T.dot(input, self.W) + self.b

        if linear:
            self.output = activation
        else:
            self.output = T.nnet.relu(activation, relu_leak)

