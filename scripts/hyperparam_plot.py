import numpy as np 
import matplotlib.pyplot as plt

logData = np.loadtxt('logs/_config_hyperparam_1run2')
nEstimators = logData[:, 0].reshape((10, 96))
accuracies = logData[:, 1].reshape((10, 96))
nEstimatorsMean = np.mean(nEstimators, axis = 0)
nEstimatorsStd = np.std(nEstimators, axis = 0)
accuraciesMean = np.mean(accuracies, axis = 0)
accuraciesStd = np.std(accuracies, axis = 0)

xAxis = np.arange(0.05, 1.01, 0.01)

plt.plot(xAxis, nEstimatorsMean, label="Mean # of estimators remaining")
plt.fill_between(xAxis, nEstimatorsMean - nEstimatorsStd, nEstimatorsMean + nEstimatorsStd, alpha = 0.3, label="Std. deviation")
plt.xlabel("Diversity trade-off")
plt.ylabel("# remaining classifiers (out of 140)")
plt.legend()
plt.figure()


plt.plot(xAxis, accuraciesMean, label="Mean accuracy")
plt.fill_between(xAxis, accuraciesMean - accuraciesStd, accuraciesMean + accuraciesStd, alpha = 0.3, label="Std. deviation")
plt.legend()

plt.xlabel("Diversity trade-off")
plt.ylabel("Accuracy")
plt.show()