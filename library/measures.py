
import numpy as np

def diversity(estimators, dataset):
    """Diversity according to the DREP (2012) paper. Goes from 0 to 2.
    """
    n_estim = len(estimators)
    if n_estim == 1:
        return 2
    return 1 - agreement(estimators, dataset) / (n_estim**2 - n_estim)

def agreement(estimators, dataset):
    """The DREP (2012) paper calls this diff, but that is highly missleading. 
    Goes from -2 to 2.
    """
    result = 0
    n_estim = len(estimators)

    # diagonal remains zero (although obviously it would produce maximum agreement)
    for i in range(n_estim):
        for j in range(i + 1, n_estim):
            result += estimatorAgreement(estimators[i], estimators[j], dataset)

    # we only calculated the upper triangular of the agreement matrix so far
    return 2 * result

def estimatorEnsembleAgreementDeprecated(estimator, ensemble, dataset):
    """agreement between an estimator and an ensemble as defined in DREP (2012), Equation (9).
    """
    return np.mean(estimator.predict(dataset) * ensemble.predict(dataset))

def estimatorEnsembleAgreement(estimator, ensemble, dataset):
    """agreement between an estimator and an ensemble as defined in DREP (2012), Equation (9).
    """
    return np.mean((estimator.predict(dataset, collapse=True) == ensemble.predict(dataset)).astype(np.int32) * 2 - 1)

def estimatorAgreementDeprecated(estimator1, estimator2, dataset):
    """agreement between two estimators on a given dataset. Goes from -1 to 1.
    DEPRECATED: Moved to multiclass generalization.
    """
    return np.mean(estimator1.predict(dataset) * estimator2.predict(dataset))

def estimatorAgreement(estimator1, estimator2, dataset):
    """agreement between two estimators on a given dataset for multiple classes. Multiplication maps disagreeing
    predictions to -1 and agreeing ones to 1. Hence, we do:
    """
    return np.mean((estimator1.predict(dataset, collapse=True) == estimator2.predict(dataset, collapse=True)).astype(np.int32) * 2 - 1)

def estimatorAgreementMatrix(estimators, dataset, withDiagonal = False):
    """agreement matrix, makes it possible to use the agreement as a matrix metric
    """
    n_estim = len(estimators)
    result = np.zeros((n_estim, n_estim))

    for i in range(n_estim):
        for j in range(n_estim):
            if i != j or withDiagonal:
                result[i, j] = estimatorAgreement(estimators[i], estimators[j], dataset)
    
    return result