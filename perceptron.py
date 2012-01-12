from optparse import OptionParser
import sys
import random

learning_rate = .2
num_features = None
train_lines = []
test_lines = []
weights = {}

def main(trainfile_string, testfile_string):
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
    # train and test each class
    for cls in weights.keys():
        train(cls)
        test(cls)
            
def train(cls):
    "Train perceptron to differentiate between cls and 8, \
    and return the trained weights weights"
    global learning_rate
    global num_features
    global train_lines

    for line in train_lines:
        # split the line of data into a vector of ints        
        vector = [int(x) for x in line.strip().split(',')]
        (o, t) = getOandT(vector, cls)
        # adjust the weights
        for i in xrange(num_features):
            weights[cls][i] += learning_rate*(t - o)*vector[i]
            
    
def test(cls):
    "Use the provided test data file to test the trained weights\
    for a given class number vs. 8"
    global test_lines
    p = 0;
    n = 0;
    for line in test_lines:
        # split the line of data into a vector of ints
        vector = [int(x) for x in line.strip().split(',')]
        (o, t) = getOandT(vector, cls)        
        if (o == t):
            p = p + 1
        else:
            n = n + 1
    print('Success Rate of %d vs. %d: %f\n' % (8, cls, float(p)/float(n + p)))

def getOandT(vector, cls):
    "Get o and t values"
    total = 0.0
    for i in xrange(num_features):
        total += weights[cls][i]*vector[i]
    o = sgn(total)
    
    # expected value of t is -1 or 1
    # an 8 is encoded as 1, and the compared number is encoded -1
    if vector[-1] == 8:
        t = 1
    else:
        t = -1
        
    return (o, t)
        
def sgn(val):
    if val > 0:
        return 1
    else:
        return -1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print 'Usage: python perceptron.py --train=<training file> --test=<test file>'
        sys.exit(1)
    else:
        parser = OptionParser()
        parser.add_option("--train", dest="trainfile",
                          help="specify the file with training data")
        parser.add_option("--test", dest="testfile",
                          help="specify the file with test data")
        (options, args) = parser.parse_args()
        main(options.trainfile, options.testfile)