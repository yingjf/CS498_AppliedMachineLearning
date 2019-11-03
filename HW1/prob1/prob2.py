import math
import pandas as pd
import numpy as np
from six.moves import urllib
from scipy.io import loadmat
from PIL import Image
from scipy.misc import imresize
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import main

def data_down():
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    mnist_path = "./mnist-original.mat"
    response = urllib.request.urlopen(mnist_alternative_url)
    with open(mnist_path, "wb") as f:
        content = response.read()
        f.write(content)
    return mnist_path

def readInMnist():
    #run this if it is the first time or you need to update
    # mnist_path = data_down()

    #since we already download the dataset, we could directly use them
    mnist_path = "./mnist-original.mat"
    mnist_raw = loadmat(mnist_path)
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
        }
    print("Success!")
    features = np.array(mnist['data']).astype(int)
    labels = np.array(mnist['target']).astype(int)
    labels = labels.reshape(1, len(labels))
    data = np.concatenate((features, labels.T), axis=1)
    return data

def splitDataset(dataset, splitRatio):
    portion = int(len(dataset)*splitRatio - 1)
    np.random.shuffle(dataset)
    trainSet, testSet = dataset[:portion, :], dataset[portion:, :]
    return [trainSet, testSet]

def getMeanandStddev(trainingSet):
    # convert numpy array to Panda dataframe
    dataset = pd.DataFrame(trainingSet)
    dataset.rename(columns={784: 'label'}, inplace=True)
    # Use Panda groupby library to get the mean
    meanGroup = dataset.groupby("label").mean()
    stddevGroup = dataset.groupby("label").std()

    return [meanGroup, stddevGroup]

def calculateProbability(x, mean, stdev):
    if (mean == 0 and stdev == 0):
        return 0
    # using Standard normal distribution
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

def calculateClassProbabilities(meanGroup, stddevGroup, inputVector, classProb):
    probabilities = {}
    for i in range(meanGroup.shape[0]):
        probabilities[i] = math.log10(classProb[i])
        for j in range(meanGroup.shape[1]):
            mean = meanGroup.iloc[i, j]
            stdev = stddevGroup.iloc[i, j]
            x = inputVector[j]
            prob = norm.pdf(x, loc=mean, scale=stdev)
            if (prob == 0 or np.isnan(prob)):
                prob = 1e-15
            probabilities[i] += math.log10(prob)
            # if (mean != 0):
            #     prob = calculateProbability(x, mean, stdev)
            #     if (prob != 0):
            #         probabilities[i] += math.log10(calculateProbability(x, mean, stdev))
    return probabilities

def getPredictions(meanGroup, stddevGroup, testSet):
    # Get probabilistic of each class in test data
    unique, counts = np.unique(testSet[:, -1], return_counts=True)
    prob = np.true_divide(counts, testSet[:, -1].shape[0])
    classProb = dict(zip(unique, prob))

    # Predict
    predictions = []
    for i in range(len(testSet)):
        probabilities = calculateClassProbabilities(meanGroup, stddevGroup, testSet[i, :], classProb)
        predictions.append(max(probabilities, key=probabilities.get))
    return predictions

def seperateFeatureClass(train, test):
    train_x = train[:, :-1]
    train_y = train[:, -1]
    test_y = test[:, -1]
    test_x = test[:, :-1]
    return [train_x, train_y, test_y, test_x]

def rescale_strech_image(image):
    x = np.array(image).reshape((-1, 1, 28, 28)).astype(np.uint8)
    img1 = Image.fromarray(x[0][0])
    bw_img = img1.point(lambda x: 0 if x < 128 else 255, '1')
    img = np.reshape(np.array(bw_img), (28, 28))
    row= np.unique(np.nonzero(img)[0])
    col = np.unique(np.nonzero(img)[1])
    image_data_new = img[min(row):max(row), min(col):max(col)]
    image_data_new = imresize(image_data_new, (20, 20))
    return (np.array(image_data_new).astype(np.uint8))

def modifyData(train_x, train_y, test_x, test_y):
    target_train_y = train_y.astype(np.uint8)
    target_test_y = test_y.astype(np.uint8)

    train_modified = np.apply_along_axis(rescale_strech_image, axis=1, arr=train_x)
    test_modified = np.apply_along_axis(rescale_strech_image, axis=1, arr=test_x)

    train_final_x = np.reshape(train_modified, (train_modified.shape[0], 400))
    test_final_x = np.reshape(test_modified, (test_modified.shape[0], 400))

    return [train_final_x, target_train_y, test_final_x, target_test_y]

def prob2():
    # read in mnist
    data = readInMnist()
    # Split dataset
    splitRatio = 0.999
    trainingSet, testSet = splitDataset(data, splitRatio)
    # Seperate the feature and classes
    train_x, train_y, test_x, test_y = seperateFeatureClass(trainingSet, testSet)
    # Train and test data with GaussianNB
    # prepare model
    meanGroup, stddevGroup = getMeanandStddev(trainingSet)
    # test model
    predictions = getPredictions(meanGroup, stddevGroup, testSet)
    accuracy_GaussianNB = main.getAccuracy(testSet, predictions)
    #
    modifyData(train_x, train_y, test_x, test_y)

prob2()
