from __future__ import print_function, division
from builtins import range, input
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from numpy import genfromtxt
import scipy
from scipy import linalg as LA


def loadCsv(filename):
    dataset = genfromtxt(filename, delimiter=',', dtype='float')
    return dataset

def obtainLowDim(dataset, e_vecs, s_val, meanDataset):
    newDataset = np.zeros(dataset.shape[1])
    # meanDataset = obtainMean(dataset)
    for i in range(0, dataset.shape[0]):
        x_i = dataset[i, :] - meanDataset
        u_vector = np.zeros(dataset.shape[1])
        for j in range(0, s_val):
            temp = np.dot(np.dot(e_vecs[:,j], x_i.T), e_vecs[:,j])
            u_vector = np.add(temp, u_vector)
        newDataset = np.vstack((newDataset, u_vector + meanDataset))

    return newDataset[1:]

def obtainMean(dataset):
    return dataset.sum(axis=0) / dataset.shape[0]

def obtainError(dataset_1, dataset_2):
    diff = np.square(dataset_1 - dataset_2)
    return diff.sum()/diff.shape[0]

def main():
    Filename1 = '../hw3-data/dataI.csv'
    Filename2 = '../hw3-data/dataII.csv'
    Filename3 = '../hw3-data/dataIII.csv'
    Filename4 = '../hw3-data/dataIV.csv'
    Filename5 = '../hw3-data/dataV.csv'
    Files = [Filename1, Filename2, Filename3, Filename4, Filename5]
    noiselessFileName = '../hw3-data/iris.csv'
    noiselessDataset = loadCsv(noiselessFileName)
    errorResult = []
    for i in range(0, len(Files)):
        noiseDataset = loadCsv(Files[i])

        convmatDataset = np.cov(noiselessDataset[1:].T)
        e_vals, e_vecs = LA.eig(convmatDataset)

        list = []

        for r in range(0, 5):
            newDataset = obtainLowDim(noiseDataset[1:], e_vecs, r, obtainMean(noiselessDataset[1:]))
            err = obtainError(newDataset, noiselessDataset[1:])
            print("Error is {} for Dataset {} with r {}, using the mean and covariance matrix of the noiseless dataset".format(err,i+1, r))
            list.append(err)
        errorResult.append(list)
        print("")
    # Write to csv file
    npArray = np.array(errorResult)
    np.savetxt('noiseless.csv', npArray, delimiter=",")

    print("")
    print("")

    errorResult = []
    for i in range(0, len(Files)):
        noiseDataset = loadCsv(Files[i])

        convmatDataset = np.cov(noiseDataset[1:].T)
        e_vals, e_vecs = LA.eig(convmatDataset)

        list = []

        for r in range(0, 5):
            newDataset = obtainLowDim(noiseDataset[1:], e_vecs, r, obtainMean(noiseDataset[1:]))
            err = obtainError(newDataset, noiselessDataset[1:])
            print("Error is {} for Dataset {} with r {}, using the mean and covariance matrix of the noisy dataset".format(err,i + 1, r))
            list.append(err)

            if r == 2 and i == 0:
                # Write to csv file
                np.savetxt('yt5-recon.csv', newDataset, delimiter=",",  header="Sepal.Length,Sepal.Width,Petal.Length ,Petal.Width")

        errorResult.append(list)
        print("")
    # Write to csv file
    npArray = np.array(errorResult)
    np.savetxt('noisy.csv', npArray, delimiter=",")

main()


