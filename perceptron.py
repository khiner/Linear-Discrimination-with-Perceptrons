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
        self.success = {}

        # initialize train_vectors and test_vectors.
        # initialize weight vectors to random values in (-1, 1)
        for n in xrange(10):
            if n != 8:
                self.train_vectors[n] = []
                self.test_vectors[n] = []
                self.weights[n] = [random.uniform(-1,1) for i in xrange(self.num_features)]
                self.success[n] = 0.0                
                for line in train_lines:
                    vector = [int(x) for x in line.strip().split(',')]
                    if vector[-1] in (n, 8):
                        self.train_vectors[n].append(vector)
                for line in test_lines:
                    vector = [int(x) for x in line.strip().split(',')]
                    if vector[-1] in (n, 8):
                        self.test_vectors[n].append(vector)
                        
    def train(self, cls, learning_rate):
        """Train perceptron to differentiate between cls and 8,
        and return the trained weights weights"""

        vectors = self.train_vectors[cls]
        for vector in vectors:
            (o, t) = self.getOandT(vector, cls)
            if o == None or t == None:
                continue
            # adjust the weights
            for i in xrange(self.num_features):
                self.weights[cls][i] += learning_rate*(t - o)*vector[i]
                
    def testAll(self, train):
        if train:
            print 'Training:'
            vectors = self.train_vectors
        else:
            print 'Testing:'
            vectors = self.test_vectors
            
        total_improve = 0.0
        for cls in self.weights.keys():
            old_success = self.success[cls]
            self.success[cls] = self.test(cls, vectors)
            improve = self.success[cls] - old_success
            total_improve = total_improve + improve
            print 'Accuracy of %d vs. %d: %f' % (cls, 8, self.success[cls])

        avg_improve = total_improve/9.0

        print 'Avg change in accuracy: %f\n' % avg_improve            
        return (avg_improve > 0)
    
    def test(self, cls, vectors):
        """Use the provided test data to test the trained weights
        for a given class number vs. 8.
        Returns success rate."""

        test_vectors = vectors[cls]
        p = 0; # positive - correct classification
        n = 0; # negative - incorrect classification
        for vector in test_vectors:
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
    # train each class, then test them all.
    # when there is no more improvement in accuracy,
    # stop training and run test
    for i in xrange(int(options.max_epochs)):
        for cls in perceptron.weights.keys():
            perceptron.train(cls, float(options.rate))
        print 'Epoch: %d' % (i + 1)
        if not perceptron.testAll(train=True):
            break
    perceptron.testAll(train=False)