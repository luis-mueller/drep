import numpy as np 
import library.measures as measures

class LabelMapClassificationEnsemble:
    """Ensemble for labels maps with majority voting that is efficiently summing one hot encodings and
    reduce the labels in one step using np.argmax.
    """
    def __init__(self, estimators = []):
        self.estimators = estimators
        self.lastVotes = None

    def copy(self):
        """Copy ensemble. 
        """
        return LabelMapClassificationEnsemble(self.estimators.copy())

    def predict(self, x):
        """Majority voting to obtain a sign for the prediction of input x. 
        """
        return self._predict(self.estimators, x)

    def score(self, X, y):
        """Computes the percentage of correctly classifed samples on a given dataset 
        """
        return self._score(self.estimators, X, y)

    def add(self, estimator):
        """Permanently add an estimator to this ensemble.  
        """
        if self.lastVotes is not None:
            self.lastVotes += estimator.predict(self.lastData)
        self.estimators.append(estimator)

    def remove(self, estimator):
        """Permanently remove an estimator from this ensemble.  
        """
        self.lastVotes = None
        self.estimators.remove(estimator)

    def scoreWith(self, estimator, X, y):
        """Compute the score of this ensemble on a dataset, assuming the given estimator was part of it. 
        """
        votes = self._getVotes(self.estimators, X)
        votes += estimator.predict(X)
        predictions = np.argmax(votes, axis = 0) + 1

        return np.mean(y == predictions)
        #return self._score(self.estimators + [estimator], X, y)

    def diversity(self, dataset):
        """Wrapper to compute diversity of ensemble given a dataset 
        """
        return measures.diversity(self.estimators, dataset)

    def moveEstimatorFromTo(estimator, fromEnsemble, toEnsemble):
        """Move estimator from one ensemble to the other.
        """
        fromEnsemble.remove(estimator)
        toEnsemble.add(estimator)

    @property 
    def nEstimators(self):
        return len(self.estimators)

    def _predict(self, estimators, x):
        votes = self._getVotes(estimators, x)
        return np.argmax(votes, axis = 0) + 1

    def _getVotes(self, estimators, x):
        if self.lastVotes is not None and np.array_equal(self.lastData, x):
            return self.lastVotes.copy()

        nClasses = len(self.estimators[0].classes)
        votes = np.zeros((nClasses, x.shape[0]))
        for estimator in estimators:
            votes += estimator.predict(x)
        self.lastVotes = votes
        self.lastData = x
        return votes

    def _score(self, estimators, X, y):
        return 0 if estimators == [] else np.mean(y == self._predict(estimators, X))


