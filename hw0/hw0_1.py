import numpy as np
import os

matrixA = []
matrixB = []

dir = os.path.abspath(os.path.dirname(__file__))
filenameA = os.path.join(dir, 'data\\matrixA.txt')
filenameB = os.path.join(dir, 'data\\matrixB.txt')
savepath = os.path.join(dir, 'data\\result.txt')

with open(filenameA) as f1:
  for l in f1.readlines():
    r = [int(x) for x in l.split(',')]
    matrixA.append(r)
  
with open(filenameB) as f1:
  for l in f1.readlines():
    r = [int(x) for x in l.split(',')]
    matrixB.append(r)
  
matrixA = np.array(matrixA)
matrixB = np.array(matrixB)

result = np.dot(matrixA, matrixB)
result.sort(axis = 1)

np.savetxt(savepath, result, fmt='%d', delimiter=',')