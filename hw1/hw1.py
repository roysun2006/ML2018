import numpy as np
import csv, os
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt

data = []
for i in range(18):
  data.append([])

dir = os.path.abspath(os.path.dirname(__file__))
trainpath = os.path.join(dir, 'data\\train.csv')
testpath = os.path.join(dir, 'data\\test.csv')
text = open(trainpath, 'r',encoding = 'big5')
traindata = csv.reader(text, delimiter=',')
n_row = 0

for r in traindata:
  if n_row != 0:
    for i in range(3, 27):
      if r[i] != 'NR':
        data[(n_row - 1) % 18].append(float(r[i]))
      else:
        data[(n_row - 1) % 18].append(float(0))
  n_row += 1
text.close()

x = []
y = []

for i in range(12):
  for j in range(471):
    x.append([])
    for t in range(18):
      for s in range(9):
        x[471*i + j].append(data[t][480* i + j + s])
    y.append(data[9][480 * i + j + 9])

x_test = []	

text = open(testpath, 'r',encoding = 'big5')
testdata = csv.reader(text, delimiter=',')
n_row = 0
for r in testdata:
  if n_row %18 == 0:
    x_test.append([])
    for i in range(2, 11):
      x_test[n_row//18].append(float(r[i]))
  else :
    for i in range(2, 11):
      if r[i] != 'NR':
        x_test[n_row // 18].append(float(r[i]))
      else:
        x_test[n_row // 18].append(float(0))
  n_row += 1
text.close()


x_test = np.array(x_test)
x = np.array(x)
y = np.array(y)


x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis = 1)

learning_rate = 1e-6
repeat = 10000
weight = np.zeros(len(x[0]))




def GradientDescent(X, W, Y, lr, round, lamb):
  cost_set = []
  for i in range(round):
    hypo = np.dot(X, W)
    loss = hypo - Y
    cost = np.sum(loss**2 / len(X))
    cost_a = np.sqrt(cost)
    cost_set.append(cost_a)
    x_t = X.transpose()
    gra = np.dot(x_t, loss) / len(X) + lamb * W
    W -= lr * gra
    if i % 500 == 0:
      print('Iteration: %d | Cost: %f  ' % (i, cost_a))
  return W, cost_set
  
def AdaGrad(X, W, Y, lr, round, lam):
  cost_set = []
  grasum = np.zeros(X.shape[1])
  for i in  range(round):
    hypo = np.dot(X, W)
    loss = hypo - Y
    cost = np.sum(loss**2 / len(X))
    cost_a = np.sqrt(cost)
    cost_set.append(cost_a)
    x_t = X.transpose()
    gra = np.dot(x_t, loss) / len(X) + lam * W
    grasum += gra ** 2
    grasq = np.sqrt(grasum)
    W -= lr * gra / grasq
    if i % 500 == 0:
      print('Iteration: %d | Cost: %f  ' % (i, cost_a))
  return W, cost_set


weight_gd, cost_gd = GradientDescent(x, weight, y, learning_rate, 10000, 50)
weight_ada, cost_ada = AdaGrad(x, weight, y, 5, 10000, 50)
weight_closeform = inv(x.T.dot(x)).dot(x.T).dot(y)

print(weight_gd)
print(weight_ada)
print(weight_closeform)

y_gd = np.dot(x_test, weight_gd)
y_ada = np.dot(x_test, weight_ada)
y_closeform = np.dot(x_test, weight_closeform)

'''
plt.plot(range(len(cost_gd)), cost_gd, "b--", "GD")
plt.plot(range(len(cost_ada)), cost_ada, "g--", "Ada")
plt.title("Training Process")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.show()
'''

plt.plot(range(len(y_gd)), y_gd, "y-", "GD")
plt.plot(range(len(y_ada)), y_ada, "g--", "Ada")
plt.plot(range(len(y_closeform)), y_closeform, "r--", "cf")
plt.show()