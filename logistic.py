
import theano, numpy, cPickle, gzip
from theano import tensor as T

IMAGE_SIZE = 28
NUM_LABELS = 10
MINIBATCH_SIZE = 1000

class LogisticRegression:

    def __init__(self, input, n_in, n_out):

        self.W = theano.shared(value = numpy.random.randn(n_in, n_out), name = 'W')
        self.b = theano.shared(value = numpy.random.randn(n_out), name = 'b')

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)

    def loss(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
    return shared_x, T.cast(shared_y, 'int32')

if __name__ == '__main__':

    f = gzip.open(r'C:\Users\alexa\DeepLearningTutorials\data\mnist.pkl.gz')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    x = T.dmatrix('x')
    y = T.ivector('y')
    index = T.iscalar('index')

    classifier = LogisticRegression(x, IMAGE_SIZE**2, NUM_LABELS)

    cost = classifier.loss(y)

    g_W = T.grad(cost, classifier.W)
    g_b = T.grad(cost, classifier.b)

    learning_rate = 0.5

    updates = [(classifier.W, classifier.W - learning_rate * g_W), (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function([index], cost, updates = updates,
        givens = { x: train_set_x[index * MINIBATCH_SIZE:(index + 1) * MINIBATCH_SIZE], y: train_set_y[index * MINIBATCH_SIZE:(index + 1) * MINIBATCH_SIZE]})

    for i in range(len(train_set[0]) / MINIBATCH_SIZE):
        c = train_model(i)

        print "Cost at iteration {}: {}".format(i, c)