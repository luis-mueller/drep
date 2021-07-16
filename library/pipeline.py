from library.reader import LabelMapIO, LabelMapDataset
from library.ensemble import LabelMapClassificationEnsemble
from library.pruning import drep

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def needsPruningData(fn):
    def wrapper(*args, **kwargs):
        if args[0].pruningData is None:
            raise ValueError('Sample pruning data first')
        return fn(*args, **kwargs)
    return wrapper


def needsPrunedEnsemble(fn):
    def wrapper(*args, **kwargs):
        if args[0].prunedEnsemble is None:
            raise ValueError('Prune ensemble first')
        return fn(*args, **kwargs)
    return wrapper


def pipelineStep(fn):
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        print("Done.")
        return res
    return wrapper


def staticInfo(fn):
    def wrapper(*args, **kwargs):
        print(fn.__doc__)
        return fn(*args, **kwargs)
    return wrapper


def argsInfo(fn):
    def wrapper(*args, **kwargs):
        print(fn.__doc__ % args)
        return fn(*args, **kwargs)
    return wrapper


class DrepPipeline:
    @pipelineStep
    @argsInfo
    def loadDB(path):
        """Loading all label maps from path %s."""
        return LabelMapIO(path).read()

    @pipelineStep
    @argsInfo
    def byDB(path):
        """Loading all label maps from path %s. and creating process."""
        estimators = LabelMapIO(path).read()
        return DrepPipeline(estimators)

    @pipelineStep
    def byModel(path):
        """Loading pruned label maps from path %s."""
        file = open(path, 'r')
        names = [f.rstrip() for f in file.readlines()]
        file.close()
        estimators = LabelMapIO.estimatorsByFiles(names)
        pipeline = DrepPipeline(estimators)
        pipeline.prunedEnsemble = pipeline.ensemble
        return pipeline

    def __init__(self, estimators):
        self.estimators = estimators
        self.ensemble = LabelMapClassificationEnsemble(self.estimators)
        self.dataset = LabelMapDataset.load()
        self.pruningData = None
        self.prunedEnsemble = None
        self.lastStats = None

    def sample(self, nPruningSamples, testSize):
        X, y = self.dataset
        if nPruningSamples > 0:
            idx = np.random.randint(0, y.shape[0], nPruningSamples)
            X, y = X[idx, :], y[idx]

        if testSize > 0:
            self.pruningData = train_test_split(X, y, test_size=testSize)
        else:
            self.pruningData = X, None, y, None
        return self

    @pipelineStep
    @needsPruningData
    @staticInfo
    def prune(self, pruningMethod, *args, **kwargs):
        """Loading pruning method and start pruning."""

        X, _, y, _ = self.pruningData
        if pruningMethod == 'drep':
            self.prunedEnsemble = drep(self.ensemble, X, y, *args, **kwargs)
        elif pruningMethod == 'identity':
            self.prunedEnsemble = self.ensemble
        else:
            raise ValueError(
                'Pruning method %s is not supported' % pruningMethod)
        return self

    @needsPruningData
    def stats(self):
        _, X, _, y = self.pruningData
        if X is not None and y is not None:
            if self.prunedEnsemble is None:
                self.__printStats(self.ensemble, X, y)
            else:
                self.__printStats(self.prunedEnsemble, X, y)
        return self

    @needsPrunedEnsemble 
    def externalValidationStats(self, path):
        validationMap = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        labelMap = self.predictEntireMap()
        contestHandIn = self.toContestFormat(labelMap)

        testIndices = np.where(validationMap != 0)
        self.lastStats = [self.prunedEnsemble.nEstimators,
                          np.mean(contestHandIn[testIndices] == validationMap[testIndices])]
        print('Number of estimators: %d' % self.lastStats[0])
        print('Accuracy of ensemble on test data: %.4f' % self.lastStats[1])

    @needsPrunedEnsemble
    def saveModel(self, path):
        LabelMapIO(path).write(self.prunedEnsemble)
        return self

    @pipelineStep
    @needsPrunedEnsemble
    @staticInfo
    def predictEntireMap(self):
        """Predicting entire label map (for the contest) and convert to appropriate
        format."""
        X = LabelMapDataset.loadEntireMap()
        return self.prunedEnsemble.predict(X).reshape((1202, 4172))

    @pipelineStep
    @staticInfo
    def toContestFormat(self, labelMap):
        """Upsampling the label map to the format required from the contest."""
        return cv2.resize(labelMap, (8344, 2404), fx=2, fy=2,
                          interpolation=cv2.INTER_NEAREST)

    @pipelineStep
    @needsPrunedEnsemble
    @staticInfo
    def predictMap(self, path):
        """Predicting the entire target image and upscales by factor 2 to make it compatible with 
        the contest format.
        """
        labelMap = self.predictEntireMap()
        contestHandIn = self.toContestFormat(labelMap)

        print("Saving output in contest format to %s" % path)
        cv2.imwrite(path, contestHandIn)
        return self

    def __printStats(self, ensemble, X, y):
        self.lastStats = [ensemble.nEstimators,
                          ensemble.score(X, y)]
        print('Number of estimators: %d' % self.lastStats[0])
        print('Accuracy of ensemble on test data: %.4f' % self.lastStats[1])
