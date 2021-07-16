import cv2 
import numpy as np 

classFile = open('data/classes.txt', 'r')
lines = classFile.readlines()

data = []
for line in lines:
    raw = line.split(":")[2].split(" ")
    lineData = [ int(r) for r in raw ]
    data.append(lineData)

np.savetxt('data/classes_rgb.txt', np.array(data, dtype=np.uint8))
