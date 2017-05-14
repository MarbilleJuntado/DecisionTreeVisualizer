"""
@author: Marbille Juntado
Copyright: 2017

This program performs Decision Tree learning on dataset provided by tic-tac-toe.data.
It is based on the ID3 algorithm. Two experiments have been performed that outputs several
files consisting of the node trace, decision tree, confusion matrix, and accuracy results.
"""

import math
import random
import sys
import plotly.plotly as py
import plotly.figure_factory as ff


class TreeNode:
    def __init__(self, name):
        """
        Construct a new 'TreeNode' object.

        :param name: The name of node
        """
        self.name = name
        self.label = None
        self.decisionAttr = None
        self.decisionGain = None
        self.decisionValue = None
        self.branches = []

    def visualizeTree(self):
        """
        Visualizes the tree 
        """
        self.visualizeTreeRecurse(0)

    def visualizeTreeRecurse(self, level):
        """
        Includes a log/trace of each node as the tree builds itself recursively
        """
        print '\t' * level + self.name,
        if self.decisionAttr and self.decisionGain:
            print 'choose ' + str(self.decisionAttr) + ' for having the highest gain of ' + str(self.decisionGain),
        if self.label:
            print ' ' + self.label,
        print '\n',
        level += 1
        for branch in self.branches:
            branch.visualizeTreeRecurse(level)

    def predictResults(self, cases, a):
        """
        Returns a list containing the predicted outcomes
        """
        predictions = []
        for c in cases:
            outcome = self.predictResultsRecurse(c, attributes)
            predictions.append(outcome)
        return predictions

    def predictResultsRecurse(self, case, a):
        """
        Recursively, method returns the predicted classification of the leaf nodes (bottom-most)
        """
        if self.name == '':

            # Leaf nodes
            if self.label == '+':
                return 'Positive'
            elif self.label == '-':
                return 'Negative'

        index = a.index(self.decisionAttr)

        if self.decisionValue == case[index]:
            return self.branches[0].predictResultsRecurse(case, a)

        if self.decisionGain:
            # Traverse to the branch where branch.decisionValue is in the case
            for b in self.branches:
                if b.decisionValue == case[index]:
                    return b.predictResultsRecurse(case, a)

def buildDTree(examples, targetAttribute, attributes):
    """
    Returns the root node of the decision tree

    :param examples: Each line of the training set
    """
    root = TreeNode('')


    #Suppose 
    if all(isPositive(example[-1]) for example in examples):
        root.label = '+'
        return root

    # Examples are all negative
    elif all(not isPositive(example[-1]) for example in examples):
        root.label = '-'
        return root

    # Attributes is empty
    elif not attributes:
        root.label = getMostCommonLabel(examples)
        return root
    else:
        result = returnAttributeHighestInfoGain(attributes, examples)
        attr = result[0]
        gain = result[1]
        attrIndex = result[2]

        root.decisionAttr = attr
        root.decisionGain = gain

        possibleValues = uniqueValues(attrIndex, examples)

        for value in possibleValues:
            newBranch = TreeNode(attr + ' = ' + value)
            newBranch.decisionAttr = attr
            newBranch.decisionValue = value
            root.branches.append(newBranch)
            branchExamples = sorted(row for row in examples if row[attrIndex] == value)

            if not branchExamples:
                leaf = TreeNode(getMostCommonValue(targetAttribute, examples, possibleValues))
                newBranch.branches.append(leaf)
            else:
                newExamples = []
                for example in branchExamples:
                    newExample = []
                    for i in range(len(example)):
                        if not i == attrIndex:
                            newExample.append(example[i])
                    newExamples.append(newExample)

                newBranch.branches.append(buildDTree(newExamples, targetAttribute, [a for a in attributes if not a == attr]))

    return root


# Determines whether a word is positive ('yes', 'true', etc.)
def isPositive(word):
    """
    Boolean function that determines whether a word is positive.
    Used in the classification of training example.
    :param word: any string
    """
    if word is not None:
        word = word.lower()
    return word == 'yes' or word == 'true' or word == 'y' or word == 't' or word == "positive" or word == '+'

def isNegative(word):
    """
    Boolean function that determines whether a word is negative.
    Used in the classification of training example.
    """
    if word is not None:
        word = word.lower()
    return word == 'no' or word == 'false' or word == 'n' or word == 'f' or word == 'negative' or word == '-'


def getMostCommonLabel(nodes):
    """
    Returns the most dominant classification in the list of nodes
    """
    pCount = 0
    nCount = 0

    for node in nodes:
        if node.label == '+':
            pCount += 1
        elif node.label == '-':
            nCount += 1

    if pCount >= nCount:
        return '+'
    else:
        return '-'

def returnAttributeHighestInfoGain(attributes, examples):
    """
    Returns the attribute with the highest information gain and the corresponding value

    :param attributes: The attributes (categories) of the data
    :param examples: The training examples from the data set
    """
    totalRows = len(examples)
    # Divide examples into positive and negative
    posExamples = sorted(row for row in examples if isPositive(row[-1]))
    negExamples = sorted(row for row in examples if not isPositive(row[-1]))

    #Calculate the info gain for the entire data set
    allExpectedInfo = infoGain(len(posExamples), len(negExamples))

    valuesGain = []

    # Calculate the entropy & gain of each attribute
    for i, attr in enumerate(attributes):

        # Skip the target attribute
        if attributes[-1] == attributes[i]:
            break

        values = uniqueValues(i, examples)

        # Info gain & probability of each value
        valuesExpectedInfo = []
        valuesProbability = []

        # Compute info gain for each value
        for value in values:
            # Count how many positive & negative examples there are for the value
            posExamplesOfValue = sorted(row for row in posExamples if row[i]==value)
            negExamplesOfValue = sorted(row for row in negExamples if row[i]==value)
            numPos = len(posExamplesOfValue)
            numNeg = len(negExamplesOfValue)
            # Compute the expected info & probability of the value & add them to the lists
            valueExpectedInfo = infoGain(numPos, numNeg)
            valueProbability = float(numPos + numNeg) / float(totalRows)
            valuesExpectedInfo.append(valueExpectedInfo)
            valuesProbability.append(valueProbability)

        # Compute entropy & gain of value and add gain to the list
        valueEntropy = entropy(valuesExpectedInfo, valuesProbability)
        valueGain = allExpectedInfo - valueEntropy
        valuesGain.append(valueGain)

    # Stores index of the attribute with the highest gain
    index = valuesGain.index(max(valuesGain))

    return [attributes[index], valuesGain[index], index]


def infoGain(count1, count2):
    """
    Returns the information gain at any particular level of tree construction
    
    :param count1: Contains the number of positively-classified training examples
    :param count2: Contains the number of negatively-classified training exampels
    """

    count1 = float(count1)
    count2 = float(count2)
    total = count1 + count2
    prob1 = count1/total
    prob2 = count2/total

    # Handles math error: log(0)
    if prob1 > 0.0 and prob2 > 0.0:
        return -prob1 * math.log(prob1, 2.0) - prob2 * math.log(prob2, 2.0)
    elif prob1 > 0.0:
        return -prob1 * math.log(prob1, 2.0)
    elif prob2 > 0.0:
        return -prob2 * math.log(prob2, 2.0)
    else:
        print 'There was an error computing information gain.'
        return 0

def entropy(p, e):
    """
    Calculates entropy

    :param p: list of probabilities for each value 
    :param e: list of information gain for each value 
    """
    entropy = 0.0
    for i in range(len(p)):
        entropy += p[i] * e[i]
    return entropy

def uniqueValues(attrIndex, examples):
    """
    Returns list of the distinct values of the current attribute
    """
    values = []
    for e in examples:
        if e[attrIndex] not in values:
            values.append(e[attrIndex])
    return values


def getMostCommonValue(attr, examples, values):
    """
    Returns the value with the highest frequency of the given attribute
    """
    valueCounts = []

    for value in values:
        valueCount = 0
        for example in examples:
            if example[attr] == value:
                valueCount += 1
        valueCounts.append(valueCount)

    maxIndex = valueCounts.index(max(valueCounts))
    return values[maxIndex]


def constructTreeFromFile(filepath):
    """
    Builds a decision tree from the training data set file
    """
    f = open(filepath, 'r')
    attrLine = f.readline()
    attributes = [a.strip() for a in attrLine.split(',')]
    examples = []
    for line in f:
        example = [item.strip() for item in line.split(',')]
        examples.append(example)

    # The last attribute is the target attribute or the classification column
    return buildDTree(examples, attributes[-1], attributes)


def parseTestCases(filepath):
    """
    Parses the test cases from the test data set file
    """
    f = open(filepath, 'r')
    cases = []
    for line in f:
        case = [item.strip() for item in line.split(',')]
        cases.append(case)

    return cases


def getAttributesFromFile(filepath):
    """
    The first line of the test file contains the attributes (categories)
    """
    f = open(filepath, 'r')
    attrLine = f.readline()
    return [a.strip() for a in attrLine.split(',')]

def gather_data(filename):
    """
    This function is used in reading the data from the original data set file
    """
    with open(filename) as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    lines = []
    for i in content:
        lines += [i.split(',')]

    return lines

def split_data(data):
    """
    Randomly divides the original data set into two equal sets:
    Training and Testing data sets

    :param: The original data sets
    """
    random.shuffle(data)
    size = len(data)
    train_data = data[:size/2]
    test_data = data[size/2:]
    return [train_data, test_data]

def header(data):
    """
    Useful for data sets without any attribute names. Generically labels
    each attribute as 'Attribute + <number>' to accommodate different datasets. 
    The last attribute is named 'Classification' (+/-).
    """
    size = len(data[0])
    headers = []
    for i in range(size):
        if i == size - 1:
            headers += ["Classification"]
        else:
            headers += ["Attribute" + str(i+1)]
    return headers


filename = raw_input("Enter filename of dataset: ")
"""
Experiment 1: Decision Tree Learning
Outputs the traces, decision tree, confusion matrix in different files
"""
predictY_actualY = 0
predictN_actualY = 0
predictY_actualN = 0
predictN_actualN = 0
for expnum in range(5):
    data = gather_data(filename)
    headers = header(data)
    new_data = split_data(data)
    train_data = new_data[0]
    test_data = new_data[1]
    with open("train_data1.txt", "w+") as myfile:
        headers = map(str, headers)
        h = ", ".join(headers)
        myfile.write(h)
        myfile.write("\n")
        for i in train_data:
            i = map(str, i)
            line = ", ".join(i)
            myfile.write(line)
            myfile.write("\n")

    with open("test_data1.txt", "w+") as myfile:
        for i in test_data:
            i = map(str, i)
            line = ", ".join(i)
            myfile.write(line)
            myfile.write("\n")

    fname = "Experiment 1 Trace " + str(expnum+1) + ".out"
    f = open(fname, "w+")
    sys.stdout = f
    trainingPath = "train_data1.txt"
    tree = constructTreeFromFile(trainingPath)
    tree.visualizeTree()
    f.close()

    tracedata = []
    with open(fname, 'r') as f:
        tracedata += f.readlines()

    tsize = len(tracedata)
    treefname = "Experiment 1 Tree " + str(expnum+1) + ".out"
    f = open(treefname, "w+")
    sys.stdout = f
    for i in range(tsize-1):
        if i%2 ==  1:
            line = tracedata[i].rstrip()
            if tracedata[i+1].strip() == '+':
                line += " (+)"
            elif tracedata[i+1].strip() == '-':
                line += " (-)"
            print line
    f.close()

    attributes = getAttributesFromFile(trainingPath)
    attributes.pop(-1)

    testingPath = "test_data1.txt"
    testCases = parseTestCases(testingPath)
    outcomes = tree.predictResults(testCases, attributes)
    testdata = []
    with open(testingPath, "r") as f:
        somedata = f.readlines()
        for i in somedata:
            testdata += [i.rstrip().split(', ')]
    testsize = len(testdata)
    linesize = len(headers)
    for i in range(testsize):
        if isPositive(outcomes[i]) and isPositive(testdata[i][linesize-1]):
            predictY_actualY += 1
        elif isNegative(outcomes[i]) and isPositive(testdata[i][linesize-1]):
            predictN_actualY += 1
        elif isPositive(outcomes[i]) and isNegative(testdata[i][linesize-1]):
            predictY_actualN += 1
        elif isNegative(outcomes[i]) and isNegative(testdata[i][linesize-1]):
            predictN_actualN += 1

matxname = "Experiment 1 Confusion Matrix.out"
f = open(matxname, "w+")
sys.stdout = f
avePYAY1 = predictY_actualY/5.0
avePNAY1 = predictN_actualY/5.0
avePYAN1 = predictY_actualN/5.0
avePNAN1 = predictN_actualN/5.0
accu1 = (avePYAY1+avePNAN1)/(len(outcomes))
print "Predicted Yes, Actual Yes: " + str(avePYAY1)
print "Predicted No, Actual No: " + str(avePNAY1)
print "Predicted Yes, Actual No: " + str(avePYAN1)
print "Predicted No, Actual No: " + str(avePNAN1)
print "Accuracy (# of Correct Predictions/Size of Test Set) = " + str(accu1)
f.close()  

"""
Experiment 2: Noisy Training Set
Similar to experiment 1 except noise has been introduced into twenty random training examples.
The confusion matrix also contains the comparison between the accuracies of experiments 1 & 2.
"""
predictY_actualY = 0
predictN_actualY = 0
predictY_actualN = 0
predictN_actualN = 0
for expnum in range(5):
    data = gather_data(filename)
    headers = header(data)
    new_data = split_data(data)
    train_data = new_data[0]
    test_data = new_data[1]
    samples = random.sample(train_data, 20)
    for i in samples:
        train_data.remove(i)
        i[len(i)-1] = "Maybe"
        train_data.insert(random.randint(1, len(train_data)), i)
    with open("train_data2.txt", "w+") as myfile:
        headers = map(str, headers)
        h = ", ".join(headers)
        myfile.write(h)
        myfile.write("\n")
        for i in train_data:
            i = map(str, i)
            line = ", ".join(i)
            myfile.write(line)
            myfile.write("\n")

    with open("test_data2.txt", "w+") as myfile:
        for i in test_data:
            i = map(str, i)
            line = ", ".join(i)
            myfile.write(line)
            myfile.write("\n")

    fname = "Experiment 2 Trace " + str(expnum+1) + ".out"
    f = open(fname, "w+")
    sys.stdout = f
    trainingPath = "train_data2.txt"
    tree = constructTreeFromFile(trainingPath)
    tree.visualizeTree()
    f.close()

    tracedata = []
    with open(fname, 'r') as f:
        tracedata += f.readlines()

    tsize = len(tracedata)
    treefname = "Experiment 2 Tree " + str(expnum+1) + ".out"
    f = open(treefname, "w+")
    sys.stdout = f
    for i in range(tsize-1):
        if i%2 ==  1:
            line = tracedata[i].rstrip()
            if tracedata[i+1].strip() == '+':
                line += " (+)"
            elif tracedata[i+1].strip() == '-':
                line += " (-)"
            print line
    f.close()

    attributes = getAttributesFromFile(trainingPath)
    attributes.pop(-1)

    testingPath = "test_data2.txt"
    testCases = parseTestCases(testingPath)
    outcomes = tree.predictResults(testCases, attributes)
    testdata = []
    with open(testingPath, "r") as f:
        somedata = f.readlines()
        for i in somedata:
            testdata += [i.rstrip().split(', ')]
    testsize = len(testdata)
    linesize = len(headers)
    for i in range(testsize):
        if isPositive(outcomes[i]) and isPositive(testdata[i][linesize-1]):
            predictY_actualY += 1
        elif isNegative(outcomes[i]) and isPositive(testdata[i][linesize-1]):
            predictN_actualY += 1
        elif isPositive(outcomes[i]) and isNegative(testdata[i][linesize-1]):
            predictY_actualN += 1
        elif isNegative(outcomes[i]) and isNegative(testdata[i][linesize-1]):
            predictN_actualN += 1

matxname = "Experiment 2 Confusion Matrix.out"
f = open(matxname, "w+")
sys.stdout = f
avePYAY2 = predictY_actualY/5.0
avePNAY2 = predictN_actualY/5.0
avePYAN2 = predictY_actualN/5.0
avePNAN2 = predictN_actualN/5.0
accu2 = (avePYAY2+avePNAN2)/(len(outcomes))
print "Predicted Yes, Actual Yes: " + str(avePYAY2)
print "Predicted No, Actual No: " + str(avePNAY2)
print "Predicted Yes, Actual No: " + str(avePYAN2)
print "Predicted No, Actual No: " + str(avePNAN2)
print "Accuracy (# of Correct Predictions/Size of Test Set) = " + str((avePYAY2+avePNAN2)/(len(outcomes)))

print "\nComparison with Experiment 1: "
print "Accuracy of Experiment 1: " + str(accu1)
print "Accuracy of Experiment 2: " + str(accu2)
f.close()  


