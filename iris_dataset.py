"""
The objective of a SVC (Support Vector Classifier) is to fit to the data from datasets of sklearn library,
returning a "best fit" hyperplane that divides, or categorizes, our data.
From there, after getting the hyperplane, we can then feed some features to our classifier
to see what the "predicted" class is.
"""

from sklearn import datasets
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

dataset = datasets.load_iris()  # load the iris datasets
data = dataset.data
target = dataset.target

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.8, random_state=42)


clf = SVC() # fit a SVM model to the data
clf.fit(X_train, y_train)

# make predictions
expected = y_test
predicted = clf.predict(X_test)

# summarize the fit of the clf
print(predicted)
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
