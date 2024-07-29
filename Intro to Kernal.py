import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import numpy as np
from sklearn.svm import LinearSVC
import pandas as pd

# Generate the dataset
# X, y = make_circles(n_samples=130, noise=0.06, factor=0.5, random_state=42)

dataset = pd.read_csv('nonLinearData.csv')

X = dataset.iloc[:, [0, 1]].values
Y = dataset.iloc[:, 2].values

plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()


Z = np.square(X[:, 0]) + np.square(X[:, 1])
squaredData = np.stack((X[:, 0], Z), axis=1)

# X **2 + Y ** 2 = Z
clf = LinearSVC(C=0.1)

clf.fit(squaredData, Y)

w = clf.coef_[0]
b = clf.intercept_[0]
slope = -w[0]/w[1]
intercept = -b/w[1]

print(
    f"Slope: {slope}, \nIntercept: {intercept}")

xAxis = np.linspace(-1, 1, 10)
yAxis = slope*xAxis + intercept
plt.figure(2)
plt.scatter(X[:, 0], Z, c=Y)
plt.plot(xAxis, yAxis, 'k')
plt.xlabel('X-Axis')
plt.ylabel('Z-Axis')
plt.show()


figure, axes = plt.subplots()
dataCircle = plt.Circle((0, 0), np.sqrt(intercept), fill=False)
plt.scatter(X[:, 0], X[:, 1], c=Y)
axes.set_aspect(1)
axes.add_artist(dataCircle)
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()
