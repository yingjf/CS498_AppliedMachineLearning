# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math

def loadCsv(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(len(dataset)):
        temp = [float(x) for x in dataset[i]]
        # values = [None if x == 0 else x for x in temp[:-1]]
        # values.append(temp[-1])
        # dataset[i] = values
        dataset[i] = temp
    return dataset

def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def mean(numbers):
    numbersNotNone = numbers
    # numbersNotNone = [x for x in numbers if x is not None]
    # if len(numbersNotNone) == 0:
    #     return 0
    return sum(filter(None, numbersNotNone)) / float(len(numbersNotNone))

def stdev(numbers):
    numbersNotNone = numbers
    # numbersNotNone = [x for x in numbers if x is not None]
    # if len(numbersNotNone) == 0:
    #     return 0
    avg = mean(numbersNotNone)
    variance = sum([pow(x - avg, 2) for x in numbersNotNone]) / float(len(numbersNotNone) - 1)
    return math.sqrt(variance)

def getMeanandStddev(dataset):
    # Seperate the dataset
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    # Obtain the mean and standard deviation for each feature
    meanAndStddevGroup = {}
    for classValue, instances in separated.items():
        summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*instances)]
        del summaries[-1]
        meanAndStddevGroup[classValue] = summaries
    return meanAndStddevGroup

def calculateProbability(x, mean, stdev):
    # using Standard normal distribution
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector, classProb):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = math.log10(classProb[classValue])
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            if x is not None:
                probabilities[classValue] += math.log10(calculateProbability(x, mean, stdev))
    return probabilities

def getPredictions(summaries, testSet):
    # SGet probabilistic of each class in test data
    classProb = {}
    for i in range(len(testSet)):
        vector = testSet[i]
        if (vector[-1] not in classProb):
            classProb[vector[-1]] = 1
        else:
            classProb[vector[-1]] += 1
    classProb = {k: v / len(testSet) for k, v in classProb.items()}
    #
    predictions = []
    for i in range(len(testSet)):
        probabilities = calculateClassProbabilities(summaries, testSet[i], classProb)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        predictions.append(bestLabel)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

def main():
    filename = 'pima-indians-diabetes.csv'
    accuracy = 0
    for i in range(10):
        # Split dataset
        splitRatio = 0.8
        dataset = loadCsv(filename)
        trainingSet, testSet = splitDataset(dataset, splitRatio)
        # prepare model
        summaries = getMeanandStddev(trainingSet)
        # test model
        predictions = getPredictions(summaries, testSet)
        accuracy += getAccuracy(testSet, predictions)
    accuracy /= 10
    print("Accuracy: {}%".format(accuracy))
