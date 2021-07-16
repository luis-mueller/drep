import cv2
import numpy as np

class LabelMap:
    def __init__(self, path):
        self.name = path
        self.img = cv2.imread(path)
        self.classes = np.loadtxt('data/classes_rgb.txt', dtype=np.uint8)
        self.oneHotPredictions = None
        self.lastData = None
    
    def predict(self, x, collapse = False):
        if self.oneHotPredictions is None or self.lastData is None or (not np.array_equal(self.lastData, x)):
            imgValue = np.expand_dims(self.img[x[:, 0], x[:, 1]], axis = 1)
            self.oneHotPredictions = np.all(self.classes == imgValue, axis = 2)
            self.lastData = x
        if collapse:
            return np.argmax(self.oneHotPredictions, axis = 1) + 1
        
        return np.transpose(self.oneHotPredictions.astype(np.int32))
    
    def score(self, X, y):
        return np.mean(y == self.predict(X, collapse=True).flatten())