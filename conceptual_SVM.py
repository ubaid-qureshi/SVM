"""
The objective of a Linear SVC (Support Vector Classifier) is to fit to the data we provide,
returning a "best fit" hyperplane that divides, or categorizes, our data.
From there, after getting the hyperplane, we can then feed some features to our classifier
to see what the "predicted" class is.
"""
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]

test = []
for i in range(len(x)):
    test.append([x[i], y[i]])  # test = [[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]]

target = [0, 1, 0, 1, 0, 1]  # also called as label or output

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(test, target)  # training machine with predefined data

print(clf.predict([5,5]))  # output [1], because plot is above the hyperplane
# 0 is printed if predicted value is below the hyperplane else 1 is printed
# 0 is predicted if [(2,2)] is given

plt.scatter(x, y)
plt.show()
