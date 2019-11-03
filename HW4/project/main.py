import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as LA
from sklearn.decomposition import PCA

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    data_labels = np.array(dict[(b'labels')])
    data_features = dict[(b'data')]
    return data_features, data_labels

def readLabels(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    batchMeta = dict[b'label_names']
    batchMeta = [x.decode('utf-8') for x in batchMeta]
    return batchMeta

def obtainMean(dataset):
    return np.mean(dataset, axis=0)

def obtainDiff(dataset_ori, dataset_transform):
   return np.mean((np.power((np.subtract(dataset_ori, dataset_transform)), 2).sum(axis=1)))

def plotMeanImage(dataset, i):
    # img = np.transpose(np.reshape(mean, (3, 32, 32)), (1, 2, 0)).astype(np.uint8)
    # plt.imshow(img)
    # plt.title(labels[i])
    # plt.savefig("meanPlots/mean_image_{0}".format(labels[i]))
    # Plot the dataset directly
    plt.plot(dataset)
    plt.xlabel("data")
    plt.ylabel("value")
    plt.title(labels[i])
    plt.savefig("meanPlots/mean_Dataset_{0}".format(labels[i]), bbox_inches='tight')
    plt.close()


def eigen(A):
    eigenValues, eigenVectors = LA.eig(A)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    labels_sorted = np.array(labels)[idx]
    return (eigenValues, eigenVectors, labels_sorted)

def obtainMDS(dataset):
    n = dataset.shape[0]
    I = np.identity(n)
    ones = np.ones(n)
    A = I - np.dot(ones, ones.T) / n
    W = np.dot(np.dot(A, dataset), A.T) * (-0.5)
    e_vals, e_vecs, labels_sorted = eigen(W)
    # plt.plot(range(0, e_vals.shape[0]), e_vals)
    # w, = np.where(e_vals > 0)
    s = 2
    L  = np.diag(np.sqrt(e_vals[:s]))
    V  = e_vecs[:,:s]
    Y  = V.dot(L)
    return Y, labels_sorted


labels = readLabels("../cifar-10-batches-py/batches.meta")
data_features_1, data_labels_1 = unpickle("../cifar-10-batches-py/data_batch_1")
data_features_2, data_labels_2 = unpickle("../cifar-10-batches-py/data_batch_2")
data_features_3, data_labels_3 = unpickle("../cifar-10-batches-py/data_batch_3")
data_features_4, data_labels_4 = unpickle("../cifar-10-batches-py/data_batch_4")
data_features_5, data_labels_5 = unpickle("../cifar-10-batches-py/data_batch_5")
data_features_6, data_labels_6 = unpickle("../cifar-10-batches-py/test_batch")
data_features = np.concatenate((data_features_1, data_features_2, data_features_3, data_features_4,
                                data_features_5, data_features_6), axis=0)
data_labels = np.concatenate((data_labels_1, data_labels_2, data_labels_3, data_labels_4,
                              data_labels_5, data_labels_6), axis=0)

data_labels_set = set(data_labels)
label_number = len(data_labels_set)

PCA = PCA(n_components=20)
errorList = []
meanList = []
eigenVectorList = []
oriImageList = []
imageList = []
for i in range(0, label_number):
    label = list(data_labels_set)[i]
    dataset_ori = data_features[np.where(data_labels == label)[0], :]
    oriImageList.append(dataset_ori)
    mean = np.mean(dataset_ori, axis=0)
    # plotMeanImage(mean, i)
    meanList.append(mean)
    dataset = dataset_ori - mean
    imageList.append(dataset)
    # dataset_new = PCA.fit_transform(dataset)
    PCA.fit(dataset)
    dataset_new = PCA.transform(dataset)  # Transform the data
    eigenVectorList.append(np.array(PCA.components_).transpose())
    dataset_transform = PCA.inverse_transform(dataset_new)
    err = obtainDiff(dataset, dataset_transform)
    errorList.append(err)

# Part A - Plot the error in bar
plt.bar(labels, errorList, align='center', alpha=0.5)
plt.xticks(range(0, len(labels)), labels, rotation='vertical')
plt.ylabel('Squared euclidean distance')
plt.title('Error between original data and the data with the first 20 principal components')
plt.savefig('error_partA.png', bbox_inches='tight')  # Save the plot to a file
plt.close()  # Close the plot instance to clear the canvas

# Part B
meanDistance = []
for i in range(0, len(meanList)):
    for j in range(0, len(meanList)):
        x = meanList[i]
        y = meanList[j]
        distance = np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y)
        meanDistance.append(distance)
# save to csv file
euclid_distance = np.array(meanDistance).reshape(10, 10)
np.savetxt("partb_distances.csv", euclid_distance, fmt='%1.2f', delimiter=',')

# --------------------- Testing ------------------------------------------------- #
# euclid_distance = np.array([[0,3935,306,10846,11051],[3935,0,4169,8811,9584],
#                            [306,4169,0,10789,10941],[10846,8811,10789,0,1156],
#                            [11051,9584,10941,1156,0]])
# labels = ["NYC", "LA", "BOSTON", "TOKYO", "SEOUL"]
# --------------------- Testing ------------------------------------------------- #

Y, labels_sorted = obtainMDS(euclid_distance)
plt.scatter(Y[:, 0], Y[:, 1])
plt.title('MDS for the distance between each class')
for i, txt in enumerate(labels):
    plt.annotate(txt, (Y[i, 0], Y[i, 1]))
plt.axis('tight')
plt.savefig('mds_partB.png')
plt.close()

# Part C
error_diff_array = []
for i in range(0, label_number):
    for j in range(0, label_number):
        temp = np.dot(np.array(eigenVectorList[j]).transpose(), np.array(imageList[i]).transpose())
        rotated_image = np.dot(np.array(eigenVectorList[j]), temp).transpose()
        reconstructed_image = np.array(rotated_image) + np.array(meanList[i])
        err = obtainDiff(oriImageList[i], reconstructed_image)
        error_diff_array.append(err)

error_matrix = (np.array(error_diff_array).reshape(label_number, label_number))
E = []
for i in range(0, label_number):
    for j in range(0, label_number):
        E.append((1 / 2) * (error_matrix[i, j] + error_matrix[j, i]))

D = np.array(E).reshape(label_number, label_number)
np.savetxt("partc_distances.csv", D, fmt='%1.2f', delimiter=',')
# Plot MDS
Y, labels_sorted = obtainMDS(D)
plt.scatter(Y[:, 0], Y[:, 1])
plt.title('MDS for the distance between each class')
for i, txt in enumerate(labels):
    plt.annotate(txt, (Y[i, 0], Y[i, 1]))
plt.axis('tight')
plt.savefig('mds_partC.png')
plt.close()