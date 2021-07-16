import cv2
import numpy as np
import os
from library.estimator import LabelMap


class LabelMapIO:
    def __init__(self, path):
        self.path = path

    def read(self):
        labelMaps = np.concatenate([
            np.core.defchararray.add(
                self.path + folder + "/", os.listdir(self.path+folder+"/"))
            for folder in os.listdir(self.path)])
        return LabelMapIO.estimatorsByFiles(labelMaps.flatten())

    def write(self, ensemble):
        names = [estimator.name + '\n' for estimator in ensemble.estimators]
        file = open(self.path, 'w')
        file.writelines(names)
        file.close()

    def estimatorsByFiles(files):
        return [LabelMap(file) for file in files]

class LabelMapDataset:
    def loadEntireMap():
        return np.concatenate(np.indices((1202, 4172)).transpose(1, 2, 0))

    def load():
        X = np.loadtxt('data/validation-image.txt', np.int)
        y = np.loadtxt('data/2018_IEEE_GRSS_DFC_GT_TR_Downscaled.txt', np.int)

        blockSize = int(4172 / 7)
        X[:, 0] += 601
        X[:, 1] += blockSize

        if X.shape[0] == y.shape[0] and X.shape[1] == 2 and len(y.shape) == 1:
            print('Dataloading successful')
        else:
            raise ValueError('Dataloading failed! Re-run data/cleandata.py')
        return X, y

    def loadUnCleaned():
        patch = (601, 2385)
        blockSize = int(4172 / 7)
        X = np.concatenate(np.indices(patch).transpose(1, 2, 0))
        X[:, 0] += 601
        X[:, 1] += blockSize

        y = cv2.imread('data/groundtruth.tif', cv2.IMREAD_GRAYSCALE)

        return X, y.flatten()
