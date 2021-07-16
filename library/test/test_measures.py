import unittest 
import measures 
import numpy as np

class DummyEstimator: 
    def __init__(self, y):
        self.y = y
    def predict(self, x):
        return self.y

class MeasuresTest(unittest.TestCase):

    def test_diversity_on_completely_agreeing_estimators(self):
        estimators = [DummyEstimator(np.ones(3)), DummyEstimator(np.ones(3))]
        dataset = "dataset_text"

        self.assertEqual(measures.diversity(estimators, dataset), 0)

    def test_agreement_on_completely_agreeing_estimators(self):
        estimators = [DummyEstimator(np.ones(3)), DummyEstimator(np.ones(3))]
        dataset = "dataset_text"

        self.assertEqual(measures.agreement(estimators, dataset), 2)

    def test_diversity_on_completely_disagreeing_estimators(self):
        estimators = [DummyEstimator(np.ones(3)), DummyEstimator(-np.ones(3))]
        dataset = "dataset_text"

        self.assertEqual(measures.diversity(estimators, dataset), 2)

    def test_agreement_on_completely_disagreeing_estimators(self):
        estimators = [DummyEstimator(np.ones(3)), DummyEstimator(-np.ones(3))]
        dataset = "dataset_text"

        self.assertEqual(measures.agreement(estimators, dataset), -2)

    def test_diversity_with_only_one_estimator(self):
        estimators = [DummyEstimator(np.ones(3))]
        dataset = "dataset_text"

        self.assertEqual(measures.diversity(estimators, dataset), 2)
