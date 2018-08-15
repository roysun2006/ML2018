import os
import numpy as np
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt

dir = os.path.abspath(os.path.dirname(__file__))
filenameA = os.path.join(dir, 'data\\lena.png')
filenameB = os.path.join(dir, 'data\\lena_modified.png')
savepath = os.path.join(dir, 'data\\result.png')

picA = imread(filenameA)
picB = imread(filenameB)

print(picA.shape)
h, w, d = picA.shape

for i in range(h):
  for j in range(w):
    if picA[i, j, :].all() == picB[i, j, :].all():
      picB[i,j, :3] = [255, 255, 255]
	
imsave(savepath, picB)	
