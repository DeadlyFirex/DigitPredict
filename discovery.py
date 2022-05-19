# Load libraries
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Define class names and import data from file
names = ['price', 'horsepower', 'luxury', 'efficiency', 'price per hp', 'weight', 'brand']
dataset = read_csv('input/data.csv', delimiter=',', names=names, dtype=None, encoding="utf-8")

# Split the data into two variables
array = dataset.values
x = array[:, 0:6]
y = array[:, 6]

# Split the data for training / validation
X_train, x_validation, Y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1, shuffle=True)

# Define models to test against
models = [('LR', LogisticRegression(solver='liblinear', multi_class='ovr')), ('LDA', LinearDiscriminantAnalysis()),
          ('KNN', KNeighborsClassifier()), ('CART', DecisionTreeClassifier()), ('NB', GaussianNB()),
          ('SVM', SVC(gamma='auto'))]

# Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean()} ({cv_results.std()})")

# Plot the result and save it to output/
pyplot.boxplot(results, labels=names)
pyplot.title("Algorithm Comparison")
pyplot.savefig("output/algorithm.pdf", dpi=1200)
pyplot.interactive(True)
pyplot.show()
