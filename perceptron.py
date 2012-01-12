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
    #!! To generalize to any classes,
    #!! this is the only code that needs to change
    for n in xrange(10):
        if n != 8:
            # start with random weights in range (-1, 1)            
            weights[n] = [random.uniform(-1,1) for i in xrange(num_features)]
    for cls in weights.keys():
        train(cls)
        test(cls)
            
def train(cls):
    "Train perceptron to differentiate between num and 8, \
    and return the trained weights weights"
    global learning_rate
    global num_features
    global train_lines

    for line in train_lines:
        # store a line of data as a list of ints
        data = [int(x) for x in line.strip().split(',')]
        t = getT(data[-1])
        o = getO(data, cls)
        # adjust the weights
        for i in xrange(num_features):
            weights[cls][i] += learning_rate*(t - o)*data[i]
            
    
def test(cls):
    global test_lines
    p = 0;
    n = 0;
    for line in test_lines:
        # store a line of data as a list of ints
        data = [int(x) for x in line.strip().split(',')]
        t = getT(data[-1])
        o = getO(data, cls)
        if (o == t):
            p = p + 1
        else:
            n = n + 1
    print('Success Rate = %f\n' % (float(p)/float(n + p)))

def getT(cls):
        # expected value must be -1 or 1    
        if cls == 8:
            return 1
        else:
            return -1

def getO(data, cls):
    "Get o value"
    total = 0.0
    for i in xrange(num_features):
        total += weights[cls][i]*data[i]
    return sgn(total)
        
def sgn(val):
    if val > 0:
        return 1
    else:
        return -1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print 'to train, please provide two filename arguments\n'
        sys.exit(1)
    else:
        main(sys.argv[1], sys.argv[2])