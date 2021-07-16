import numpy as np 
import cv2 
import argparse

def accuracy(source, target):
    source = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(target, cv2.IMREAD_GRAYSCALE)

    testIndices = np.where(source != 0)
    print(np.mean(target[testIndices] == source[testIndices]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Outputs the classification accuracy of one label map by the other.""")
    parser.add_argument('source', type=str,
                        help='reference .tif')

    parser.add_argument('target', type=str,
                        help='target .tif')

    args = parser.parse_args()
    accuracy(args.source, args.target)