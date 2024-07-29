from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

"""# ------ Hard And Soft Margin (How Soft Margin Can Out Perform Hard Margin)
dataset = pd.read_csv('data.csv') # This Dataset has outliers in it (added manually)
X = dataset.iloc[:, [0, 1]].values
Y = dataset.iloc[:, 2].values

xTest, yTest = make_blobs(n_samples=350, centers=2,
                          random_state=42, cluster_std=3.5)
# ---------
"""

xData, yData = make_blobs(n_samples=350, centers=2,
                          random_state=42, cluster_std=3.5)
X, xTest, Y, yTest = train_test_split(xData, yData, test_size=0.5)


clf = LinearSVC(C=0.01)
clf.fit(X, Y)


# ax1 + bx2 + c = 0
coef = clf.coef_[0]
intercept = clf.intercept_

# compute the slope and y-intercept of the decision boundary
# x2 = -a/b * x1 - c/b
slope = -coef[0]/coef[1]
y_intercept = -intercept / coef[1]


# compute the y-intercept of the two Hyperplanes
# ax1 + bx2 + c = 1
# x2 = -a/b * x1 + (-c + 1) / b
y_intercept1 = (-intercept + 1) / coef[1]


# ax1 + bx2 + c = -1
# x2 = -a/b * x1 + (-c - 1) / b
y_intercept2 = (-intercept - 1) / coef[1]

margin = 2 / np.sqrt(np.sum(coef ** 2))

print(
    f"Slope: {slope}, \nIntercept: {y_intercept} \nMargin: {margin}\nModel Score: {clf.score(xTest, yTest)}")


yPred = clf.predict(xTest)
confusionMatrix = confusion_matrix(yTest, yPred)

print("-"*38 +
      f'\nTrue Positive: {confusionMatrix[0][0]} | False Positive: {confusionMatrix[0][1]}')
print("-"*38 +
      f'\nTrue Negative: {confusionMatrix[1][1]}  | False Negative: {confusionMatrix[1][0]}\n' + "-"*38)


# 2 Values On X-Axis
xAxis = np.linspace(X[:, 0].min(), X[:, 0].max(), 2)

# Calculate The Y Values For The Above X Values
# Equation of Decision Boundary: x2 = -a/b * x1 - c/b
yAxis = slope*xAxis + y_intercept
# Now We Have Two Points That Lie On The Decision Boundary

# Equations of Hyperplanes
upHyperplane = slope*xAxis + y_intercept1
downHyperplane = slope*xAxis + y_intercept2


# plot the Training data
plt.scatter(X[:, 0], X[:, 1], c=Y)

# Plot the Decision Boundary
plt.plot(xAxis, yAxis, 'k')

# plot the Hyperplanes
plt.plot(xAxis, upHyperplane, 'k--')
plt.plot(xAxis, downHyperplane, 'k--')
plt.xlim(-10, 10)
plt.ylim(-5, 17)
plt.title("SVM Soft / Hard Margin")
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.show()


# Plot the Predected Values for Testing Data
plt.figure(1)
plt.scatter(xTest[:, 0], xTest[:, 1], c=yPred)
plt.plot(xAxis, yAxis, 'k-')
plt.xlim(-10, 10)
plt.ylim(-5, 17)
plt.title("Predicated Data")
plt.xlabel("Feature1")
plt.ylabel("Feature2")


# Plot the Testing Data
plt.figure(2)
plt.scatter(xTest[:, 0], xTest[:, 1], c=yTest)
plt.plot(xAxis, yAxis, 'k-')
plt.xlim(-10, 10)
plt.ylim(-5, 17)
plt.title("Testing Data")
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.show()
