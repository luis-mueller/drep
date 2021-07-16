import numpy as np
from library.ensemble import LabelMapClassificationEnsemble
import library.measures as measures


def drep(ensemble, X, y, diversityTradeOff=0.2):
    """DREP method from "Diversity Regularized Ensemble Pruning" (Li et. al 2012).

    X, y: Validation dataset
    diversityTradeOff: rho from the paper. The lower the value the more estimators are selected in terms of diversity optimization.
    """
    print(diversityTradeOff)

    ensemble = ensemble.copy()
    prunedEnsemble = LabelMapClassificationEnsemble(estimators=[])
    score = np.inf
    bestPerformer, _ = bestPerformingEstimatorFrom(ensemble.estimators, X, y)

    while prunedEnsemble.score(X, y) < score:
        LabelMapClassificationEnsemble.moveEstimatorFromTo(
            bestPerformer, ensemble, prunedEnsemble)

        if ensemble.nEstimators == 0:
            break

        print("Size of pruned ensemble %d" % prunedEnsemble.nEstimators)

        # Find the estimator among the ones which increase diversity most which increases the score most
        bestPerformer = bestCombinedPerformingEstimator(
            mostDiverseEstimators(
                ensemble, prunedEnsemble, X, diversityTradeOff),
            prunedEnsemble, X, y
        )

        score = prunedEnsemble.scoreWith(bestPerformer, X, y)

    return prunedEnsemble

def mostDiverseEstimators(ensemble, prunedEnsemble, dataset, diversityTradeOff):
    agreementValues = [measures.estimatorEnsembleAgreement(
        estimator, prunedEnsemble, dataset) for estimator in ensemble.estimators]
    cutoff = int(np.ceil(diversityTradeOff * ensemble.nEstimators))
    return [ensemble.estimators[i] for i in np.argsort(agreementValues)][:cutoff]


def bestPerformingEstimatorFrom(estimators, X, y):
    scores = [estimator.score(X, y) for estimator in estimators]
    index = np.argmax(scores)
    print(scores[index])
    return estimators[index], index


def bestCombinedPerformingEstimator(estimators, prunedEnsemble, X, y):
    index = np.argmax([prunedEnsemble.scoreWith(estimator, X, y) for estimator in estimators])
    return estimators[index]