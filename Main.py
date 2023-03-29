import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
# from sklearn.metrics import mean_absolute_error

# Training data
trainingData = pd.read_csv('C:/Users/Tobia/Desktop/csvs/Training.csv')
trainingData.drop('Unnamed: 133', axis=1, inplace=True)  # drop af tom column i slutningen af dataen

X = trainingData
y = trainingData["prognosis"]
X.drop(['prognosis'], axis=1, inplace=True)

# Test data
testData = pd.read_csv('C:/Users/Tobia/Desktop/csvs/Testing.csv')

XTest = testData
yTest = XTest.prognosis
XTest.drop(['prognosis'], axis=1, inplace=True)

# Prøver at splittet training sættet, da der er flere rows i den
splitTrainX, splitPredX, splitTrainY, splitPredY = train_test_split(X, y, random_state=1)

# DecisionTreeClassifier
dtModel = DecisionTreeClassifier(random_state=1)
dtModel.fit(X, y)
dtPred = dtModel.predict(XTest)
print("DT Accuracy: {:.3f}%".format(metrics.accuracy_score(yTest, dtPred)*100))

# RandomForestRegressor
rfmodel = RandomForestClassifier(random_state=1)
rfmodel.fit(X, y)
rfPred = rfmodel.predict(XTest)
print("RF Accuracy: {:.3f}%".format(metrics.accuracy_score(yTest, rfPred)*100))

# RFC, men med training dataen splittet i to
rfmodel.fit(splitTrainX, splitTrainY)
rfPred = rfmodel.predict(splitPredX)
print("RF training split Accuracy: {:.3f}%".format(metrics.accuracy_score(splitPredY, rfPred)*100))

# Logistic regression? <- "It is the go-to method for binary classification problems"
# Support vector machine? <- bruger man vidst ikke mere i følge Simon
# Naïve bayes algorithm?
# K-nearest neighbour


# columns = []
# for c in X.columns:
#    columns.append(c)
