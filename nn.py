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