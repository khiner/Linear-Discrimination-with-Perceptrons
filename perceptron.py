from optparse import OptionParser
import random

class Perceptron(object):
    def __init__(self, trainfile_string, testfile_string):
        train_lines = open(trainfile_string, 'r').readlines()
        test_lines = open(testfile_string, 'r').readlines()
        # these vectors map a class number to a list of its corresponding
        # train/test instances (int vectors)
        self.train_vectors = {}
        self.test_vectors = {}
        self.num_features = len(train_lines[0].strip().split(',')) - 1
        self.weights = {}

        # initialize train_vectors and test_vectors.
        # initialize weight vectors to random values in (-1, 1)
        for n in xrange(10):
            if n != 8:
                self.train_vectors[n] = []
                self.test_vectors[n] = []
                self.weights[n] = [random.uniform(-1,1) for i in xrange(self.num_features)]
                for line in train_lines:
                    vector = [int(x) for x in line.strip().split(',')]
                    if vector[-1] in (n, 8):
                        self.train_vectors[n].append(vector)
                for line in test_lines:
                    vector = [int(x) for x in line.strip().split(',')]
                    if vector[-1] in (n, 8):
                        self.test_vectors[n].append(vector)

    def run(self, max_epochs, rate):
        """For each class, train until there is no more improvement
        (or accuracy is a perfect 1.0), then test the class using the
        test file"""
        
        for cls in self.weights.keys():
            improvement = 1
            epoch = 0
            accuracy = 0
            print '_%dv%d_\nTraining:' % (cls, 8)
            while accuracy < 1.0 and \
                  improvement > 0 and epoch < max_epochs:
                epoch = epoch + 1
                self.train(cls, rate)
                (old_accuracy, accuracy) = (accuracy, self.test(cls, train=True))
                improvement = accuracy - old_accuracy
                print 'Epoch %d, accuracy: %f, improvement: %f' % \
                (epoch, accuracy, improvement)
            print 'Testing:\naccuracy: %f\n' % \
            self.test(cls, train=False)
        
    def train(self, cls, learning_rate):
        """Train perceptron to differentiate between cls and 8,
        and return the trained weights weights"""

        for vector in self.train_vectors[cls]:
            (o, t) = self.getOandT(vector, cls)
            if o == None or t == None:
                continue
            # adjust the weights
            for i in xrange(self.num_features):
                self.weights[cls][i] += learning_rate*(t - o)*vector[i]

    def test(self, cls, train):
        """Use the provided test data to test the trained weights
        for a given class number vs. 8.
        Returns accuracy rate."""

        if train:
            vectors = self.train_vectors[cls]
        else:
            vectors = self.test_vectors[cls]
        
        p = 0; # positive - correct classification
        n = 0; # negative - incorrect classification
        for vector in vectors:
            (o, t) = self.getOandT(vector, cls)
            if o == None or t == None:
                continue
            elif o == t:
                p = p + 1
            else:
                n = n + 1

        return float(p)/float(n + p)

    def getOandT(self, vector, cls):
        """Returns a tuple of o and t values, comparing cls to 8
        -1 = cls, 1 = 8
        None = provided instance vector is not 8 or the provided class (cls)"""    
        if vector[-1] == 8:
            t = 1
        elif vector[-1] == cls:
            t = -1
        else:
            return (None, None)
        
        total = 0.0
        for i in xrange(self.num_features):
            total += self.weights[cls][i]*vector[i]
            o = sgn(total)
        
        return (o, t)
    
def sgn(val):
    """Returns 1 for val > 0 and -1 for val <= 0"""
    if val > 0:
        return 1
    else:
        return -1

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-n", "--train", dest="train_file", default="data/optdigits.tra", help="file with training data. default: 'data/optdigits.tra'")
    parser.add_option("-t", "--test", dest="test_file", default="data/optdigits.tes", help="file with test data. default: 'data/optdigits.tes'")
    parser.add_option("-e", "--epochs", dest="max_epochs", default=10, help="maximum number of epochs.  default: 10")
    parser.add_option("-r", "--rate", dest="rate", default=.2, help="learning rate.  default: 0.2")
    
    (options, args) = parser.parse_args()
    perceptron = Perceptron(options.train_file, options.test_file)
    perceptron.run(max_epochs=int(options.max_epochs), \
                   rate=float(options.rate))
    