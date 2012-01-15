from optparse import OptionParser
import sys
import random

class Perceptron(object):
    def __init__(self, trainfile_string, testfile_string):
        self.train_lines = open(trainfile_string, 'r').readlines()
        self.test_lines = open(testfile_string, 'r').readlines()    
        self.num_features = len(self.train_lines[0].strip().split(',')) - 1
        self.weights = {}
        self.success = {}
        
        # initialize weight vectors to random values for each class.
        for n in xrange(10):
            if n != 8:
                # start with random weights in range (-1, 1)            
                self.weights[n] = [random.uniform(-1,1) for i in xrange(self.num_features)]
                self.success[n] = 0.0

    def train(self, cls, learning_rate):
        """ Train perceptron to differentiate between cls and 8,
        and return the trained weights weights """
        
        for line in self.train_lines:
            # split the line of data into a vector of ints    
            vector = [int(x) for x in line.strip().split(',')]
            (o, t) = self.getOandT(vector, cls)
            if o == None or t == None:
                continue
            # adjust the weights
            for i in xrange(self.num_features):
                self.weights[cls][i] += learning_rate*(t - o)*vector[i]
                
    def testAll(self, train):
        if train:
            print 'Training:'
        else:
            print 'Testing:'        
        total_improve = 0.0
        for cls in self.weights.keys():
            old_success = self.success[cls]
            self.success[cls] = self.test(cls, train)
            improve = self.success[cls] - old_success
            total_improve = total_improve + improve
            print 'Accuracy of %d vs. %d: %f' % (cls, 8, self.success[cls])

        avg_improve = total_improve/9.0

        print 'Avg change in accuracy: %f\n' % avg_improve            
        return (avg_improve > 0)
    
    def test(self, cls, train):
        """ Use the provided test data file to test the trained weights
        for a given class number vs. 8.
        If train is true, testing is done using the training file.
        Returns success rate. """

        if train:
            lines = self.train_lines
        else:
            lines = self.test_lines
            
        p = 0; # positive - correct classification
        n = 0; # negative - incorrect classification
        for line in lines:
            # split the line of data into a vector of ints
            vector = [int(x) for x in line.strip().split(',')]
            (o, t) = self.getOandT(vector, cls)
            if o == None or t == None:
                continue
            elif o == t:
                p = p + 1
            else:
                n = n + 1

        return float(p)/float(n + p)

    def getOandT(self, vector, cls):
        """ Returns a tuple of o and t values, comparing cls to 8
        -1 = cls
        1 = 8
        None = provided instance vector is not 8 or the provided class (cls) """    
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
    """ Returns 1 for positive (> 0) and -1 for <= 0 """
    if val > 0:
        return 1
    else:
        return -1

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--train", dest="trainfile", default="data/optdigits.tra", help="file with training data. default is 'data/optdigits.tra'.")
    parser.add_option("--test", dest="testfile", default="data/optdigits.tes", help="file with test data. default is 'data/optdigits.tes'.")
    parser.add_option("--max_epochs", dest="max_epochs", default=10, help="maximum number of epochs.  default is 10.")
    parser.add_option("--rate", dest="rate", default=.2, help="learning rate.  default is .2")
    
    (options, args) = parser.parse_args()
    perceptron = Perceptron(options.trainfile, options.testfile)
    # train each class, then test them all.
    # when there is no improvement, break
    for i in xrange(int(options.max_epochs)):
        for cls in perceptron.weights.keys():
            perceptron.train(cls, float(options.rate))
        print 'Epoch: %d' % (i + 1)
        if not perceptron.testAll(train=True):
            break
    perceptron.testAll(train=False)