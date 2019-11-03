import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as LA
from sklearn.decomposition import PCA
from scipy.cluster.vq import kmeans, vq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import os

def get_accuracy(prediction, actual):
    return float(np.sum(prediction == actual) / prediction.shape[0])

def get_histograms(clx, n_cluster):
    feature = []
    unique, counts = np.unique(clx, return_counts=True)
    dictionary = dict(zip(unique, counts))
    for i in range(n_cluster):
        if i in dictionary.keys():
            feature.append(float(dictionary[i] / clx.shape[0]))
        else:
            feature.append(0)
    return feature

def readFiles(dirPath, n_cluster, n_sample, overlap):
    file_data = np.genfromtxt(dirPath, delimiter=' ', dtype=float)
    data = generate_segments(file_data, n_cluster, n_sample, overlap)

    return data

def generate_features(data, n_cluster, n_sample, dataPath, ratio, overlap):
    # computing K-Means with K = n_cluster
    centroids, _ = kmeans(data, n_cluster)
    features_train = []
    features_test = []
    for subdir, dirs, files in os.walk(dataPath):
        classes = dirs
        print(dirs)
        ind = 0
        for folder in dirs:
            dirs = os.listdir(os.path.join(dataPath, folder))
            numFiles = len(dirs)
            for i in range(0, int(numFiles * ratio)):
                train_data = readFiles(os.path.join(dataPath, folder, dirs[i]), n_cluster, n_sample, overlap)
                # assign each sample to a cluster
                clx_train, _ = vq(train_data, centroids)
                # calculate histograms of cluster centers
                feature = get_histograms(clx_train, n_cluster)

                # Plot histogram
                plt.bar(np.arange(len(feature)), feature, align='center', alpha=0.5, color='k')
                plt.xticks([0, 50, 100, 150, 200])
                plt.ylabel('Probability')
                plt.title(classes[ind])
                plt.savefig('histogram/' + classes[ind] + '.png', bbox_inches='tight')  # Save the plot to a file
                plt.close()  # Close the plot instance to clear the canvas

                # store feature values
                feature.append(ind)
                features_train.append(feature)

            for j in range(int(numFiles * ratio), numFiles):
                test_data = readFiles(os.path.join(dataPath, folder, dirs[j]), n_cluster, n_sample, overlap)
                # assign each sample to a cluster
                clx_test, _ = vq(test_data, centroids)
                # calculate histograms of cluster centers
                feature = get_histograms(clx_test, n_cluster)
                feature.append(ind)
                features_test.append(feature)
            ind += 1

    return np.array(features_train), np.array(features_test), classes

def generate_segments(data, n_cluster, n_sample, overLap):
    no_of_rows = np.shape(data)[0]
    no_of_col = np.shape(data)[1]
    no_overlap = int(n_sample * overLap)
    data_overlap = np.zeros(no_of_col * n_sample).reshape(1, -1)
    ind = 0
    while ind + n_sample < no_of_rows:
        data_vector = data[ind:ind + n_sample, :].reshape(1, -1)
        data_overlap = np.concatenate((data_overlap, data_vector), axis=0)
        ind += no_overlap
    return data_overlap[1:]

def readFolder(folder, n_cluster, n_sample, ratio, overlap):
    dirs = os.listdir(folder)
    numFiles = len(dirs)
    # Read in training data
    train_data = np.zeros(3).reshape(1, 3)
    for i in range(0, int(numFiles * ratio)):
        file_data = np.genfromtxt(os.path.join(folder, dirs[i]), delimiter=' ', dtype=float)
        train_data = np.concatenate((np.array(train_data), file_data), axis=0)
    train_data = train_data[1:]
    train_data = generate_segments(train_data, n_cluster, n_sample, overlap)

    # Read in testing data
    test_data = np.zeros(3).reshape(1, 3)
    if ratio != 1:
        for i in range(int(numFiles * ratio), numFiles):
            file_data = np.genfromtxt(os.path.join(folder, dirs[i]), delimiter=' ', dtype=float)
            test_data = np.concatenate((test_data, file_data), axis=0)
        test_data = test_data[1:]
        test_data = generate_segments(test_data, n_cluster, n_sample, overlap)

    return train_data, test_data

n_cluster_list = [120, 240, 480] # set the no of cluster
overlap_list = [0.1, 0.5, 1]
n_sample_list = [24, 32, 40] # set the no of samples per segment
# for n_cluster in n_cluster_list:
#     for overlap in overlap_list:
#         for n_sample in n_sample_list:

# ------ Given values -----
n_cluster = 240
overlap = 0.1
n_sample = 32
# ------ Given values -----

#
ratio = float(2 / 3)
dataPath = "../HMP_Dataset"
dirs = os.listdir(dataPath)
train_vectors = np.zeros(n_sample * 3).reshape(1, n_sample * 3)
test_vectors = np.zeros(n_sample * 3).reshape(1, n_sample * 3)
for subdir, dirs, files in os.walk(dataPath):
    for folder in dirs:
        train_data, test_data = readFolder(os.path.join(dataPath, folder), n_cluster, n_sample, 1, overlap)
        train_vectors = np.concatenate((train_vectors, train_data), axis=0)
# Make features by using K means
features_train, features_test, classes = generate_features(train_vectors[1:], n_cluster, n_sample, dataPath, ratio, overlap)
# classify using random forest classifier
clf = RandomForestClassifier(n_estimators=90, max_depth=32, random_state=8)
clf.fit(features_train[:, :n_cluster], features_train[:, n_cluster])
prediction = clf.predict(features_test[:, :n_cluster])
acc = get_accuracy(np.array(prediction), features_test[:, n_cluster])
print("The accuracy is {}% when using K: {}, overlap: {}, n_sample: {}.".format(acc * 100, n_cluster, overlap, n_sample))
print(classes)
print(confusion_matrix(features_test[:, n_cluster], prediction))
