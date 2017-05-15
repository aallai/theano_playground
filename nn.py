import theano
import numpy

INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 32
SCREEN_BUFFER_SIZE = 84
FILTER_SIZE = 8
FILTER_STRIDE = 4
BATCH_SIZE = 1

#filters = [[numpy.random.randn(FILTER_SIZE, FILTER_SIZE)] for i in range(OUTPUT_CHANNELS)]

#input = numpy.random.randn(SCREEN_BUFFER_SIZE * SCREEN_BUFFER_SIZE)

input = theano.tensor.tensor4('input')
filters = theano.tensor.tensor4('filters')

convnet = theano.tensor.nnet.conv2d(
    input, filters,
    input_shape = (BATCH_SIZE, INPUT_CHANNELS, SCREEN_BUFFER_SIZE, SCREEN_BUFFER_SIZE),
    filter_shape = (OUTPUT_CHANNELS, INPUT_CHANNELS, FILTER_SIZE, FILTER_SIZE),
    border_mode = 'valid', subsample=(FILTER_STRIDE, FILTER_STRIDE))

class ReluConv:

    # Assumes square input.
    def __init__(self, input, input_size, filter_size, stride, input_channels, output_channels, relu_leak):


        self.filters = theano.shared(value = [[numpy.random.randn(filter_size, filter_size) for i in range(input_channels)] for i in range(output_channels)], borrow = True)

        self.convnet = theano.tensor.nnet.conv2d(input, self.filters, border_mode = 'valid', subsample = (stride, stride))

        self.output = theano.tensor.relu(self.convnet, relu_leak)