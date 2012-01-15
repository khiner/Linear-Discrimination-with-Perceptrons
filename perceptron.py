from optparse import OptionParser
import sys
import random

num_features = None
train_lines = []
test_lines = []
weights = {}

def main(trainfile_string, testfile_string, epochs, learning_rate):
    global train_lines
    global test_lines
    global num_features
    global weights
    
    train_lines = open(trainfile_string, 'r').readlines()
    test_lines = open(testfile_string, 'r').readlines()    
    num_features = len(train_lines[0].strip().split(',')) - 1

    # initialize weight vectors to random values for each class.
    for n in xrange(10):
        if n != 8:
            # start with random weights in range (-1, 1)            
            weights[n] = [random.uniform(-1,1) for i in xrange(num_features)]
    # train each class for the specified num of epochs, and test each
    for cls in weights.keys():
        for i in xrange(epochs):
            train(cls, learning_rate)
        test(cls)
            
def train(cls, learning_rate):
    """
    Train perceptron to differentiate between cls and 8,
    and return the trained weights weights
    """
    global num_features
    global train_lines

    for line in train_lines:
        # split the line of data into a vector of ints    
        vector = [int(x) for x in line.strip().split(',')]
        (o, t) = getOandT(vector, cls)
        if o == None or t == None:
            continue
        # adjust the weights
        for i in xrange(num_features):
            weights[cls][i] += learning_rate*(t - o)*vector[i]
            
    
def test(cls):
    """
    Use the provided test data file to test the trained weights
    for a given class number vs. 8
    """
    global test_lines
    p = 0; # positive - correct classification
    n = 0; # negative - incorrect classification
    for line in test_lines:
        # split the line of data into a vector of ints
        vector = [int(x) for x in line.strip().split(',')]
        (o, t) = getOandT(vector, cls)
        if o == None or t == None:
            continue
        elif o == t:
            p = p + 1
        else:
            n = n + 1
    print 'Success Rate of %d vs. %d: %f\n' % (cls, 8, float(p)/float(n + p))

def getOandT(vector, cls):
    """
    returns a tuple of o and t values, comparing cls to 8
    -1 = cls
    1 = 8
    None = provided instance vector is not 8 or the provided class (cls)
    """
    if vector[-1] == 8:
        t = 1
    elif vector[-1] == cls:
        t = -1
    else:
        return (None, None)
    
    total = 0.0
    for i in xrange(num_features):
        total += weights[cls][i]*vector[i]
    o = sgn(total)
            
    return (o, t)
        
def sgn(val):
    """
    Sign function returns 1 for positive (> 0)
    or -1 for <= 0
    """
    if val > 0:
        return 1
    else:
        return -1

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--train", dest="trainfile", default="data/optdigits.tra", help="file with training data. default is 'data/optdigits.tra'.")
    parser.add_option("--test", dest="testfile", default="data/optdigits.tes", help="file with test data. default is 'data/optdigits.tes'.")
    parser.add_option("--epochs", dest="epochs", default=5, help="number of epochs.  default is 5.")
    parser.add_option("--rate", dest="rate", default=.2, help="learning rate.  default is .2")
    
    (options, args) = parser.parse_args()
    print options.trainfile + options.testfile
    main(options.trainfile, options.testfile, int(options.epochs), float(options.rate))