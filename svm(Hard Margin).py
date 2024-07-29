from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt


# generate some random data
X, y = make_blobs(n_samples=100, centers=2, random_state=42)

# train a LinearSVC model
clf = LinearSVC(C=0.01)
clf.fit(X, y)


# get the coefficient and intercept of the decision boundary
# Eqn: ax + by + c = 0
# wT * X + b = 0
coef = clf.coef_[0]
intercept = clf.intercept_


# compute the slope and y-intercept of the main Margin (Middle One)
# y = -a/bx - c/b
slope = -coef[0]/coef[1]
y_intercept = -intercept / coef[1]


# compute the y-intercept of the two margins
y_intercept1 = (-intercept - 1) / coef[1]
y_intercept2 = (-intercept + 1) / coef[1]

margin = 1 / np.sqrt(np.sum(coef ** 2))

print(
    f"Slope: {slope}, \nIntercept: {y_intercept} \nMargin: {2*margin}")

# plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y)

xAxis = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)

# Equation of the Hyperplane
yAxis = slope*xAxis + y_intercept

# Equations of Margins' Lines
upMargin = slope*xAxis + y_intercept1
downMargin = slope*xAxis + y_intercept2

# Plot the Hyperplane
plt.plot(xAxis, yAxis, 'k-')

# plot the Margins
plt.plot(xAxis, upMargin, 'k--')
plt.plot(xAxis, downMargin, 'k--')
plt.xlim(-10, 10)
plt.ylim(-5, 17)
plt.title("SVM Hard Margin")
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.show()
