import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn import metrics

from sklearn.preprocessing import OrdinalEncoder

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

# DecisionTreeClassifier
dtModel = DecisionTreeClassifier(random_state=1)
dtModel.fit(X, y)
dtPred = dtModel.predict(XTest)

print("Accuracy: {:.2f}%".format(metrics.accuracy_score(yTest, dtPred)*100))

# RandomForestRegressor
rfmodel = RandomForestRegressor(random_state=1)
rfmodel.fit(X, y)

ordinalEncoder = OrdinalEncoder()
label_X_train = X.copy()
label_X_test = XTest.copy()

predictions = rfmodel.predict(XTest)
mae = mean_absolute_error(predictions, yTest)

print("MAE: {}".format(mae))


# columns = []
# for c in X.columns:
#    columns.append(c)
