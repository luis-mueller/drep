import cv2
import numpy as np

filterCond = lambda cond: np.transpose(np.nonzero(cond > 0))

# Load the original groundtruth 
groundtruth = cv2.imread('data/2018_IEEE_GRSS_DFC_GT_TR.tif', cv2.IMREAD_GRAYSCALE)
y = cv2.resize(groundtruth, (2384, 601), fx=0.5, fy=0.5, interpolation = cv2.INTER_NEAREST)

# Save groundtruth for further processing 
cv2.imwrite('data/2018_IEEE_GRSS_DFC_GT_TR_Downscaled.tif', y)

# Throw out training samples (magic numbers are 11 and 76)
# 1 => not used in training
trainingSamples = cv2.imread('data/trainSamples.png', cv2.IMREAD_GRAYSCALE) == 11

# 1 => not used in training and not invalid
validationImage = (y * trainingSamples[601:, 596:(596+2384)] > 0)

cv2.imwrite('data/validation-image.tif', validationImage * 255)
cv2.imwrite('data/validation-image.png', validationImage * 255)

# Save labels synchronized with cleaned data
np.savetxt('data/2018_IEEE_GRSS_DFC_GT_TR_Downscaled.txt', y[np.nonzero(validationImage)], '%d')

# Save to disk
validationImage = filterCond(validationImage)
np.savetxt('data/validation-image.txt', validationImage, '%d')
print("Successfull? " + 'Yes!' if np.array_equal(np.loadtxt('data/validation-image.txt', np.int), validationImage) else 'No...')