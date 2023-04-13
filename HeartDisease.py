import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder


# https://www.kaggle.com/datasets/ambujdevsingh/key-indicators-of-heart-disease
# Explanation of the variables of the dataset
#
# Heart Disease: Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI).
# BMI : Body Mass Index (BMI).
# Smoking : Have you smoked at least 100 cigarettes in your entire life? ( The answer Yes or No ).
# AlcoholDrinking : Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week
# Stroke : (Ever told) (you had) a stroke?
# PhysicalHealth : Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? (0-30 days).
# MentalHealth : Thinking about your mental health, for how many days during the past 30 days was your mental health not good? (0-30 days).
# DiffWalking : Do you have serious difficulty walking or climbing stairs?
# Sex : Are you male or female?
# AgeCategory: Fourteen-level age category.
# Race : Imputed race/ethnicity value.
# Diabetic : (Ever told) (you had) diabetes?
# PhysicalActivity : Adults who reported doing physical activity or exercise during the past 30 days other than their regular job.
# GenHealth : Would you say that in general your health is…
# SleepTime : On average, how many hours of sleep do you get in a 24-hour period?
# Asthma : (Ever told) (you had) asthma?
# KidneyDisease : Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?
# SkinCancer : (Ever told) (you had) skin cancer?

data = pd.read_csv('C:/Users/Emil9/Datasets/heart_2022_Key_indicators.csv')
# For at se om der er nogle columns uden values. (Det var der ikke)
# print(data.isna().sum())
X = data
y = data['HeartDisease']
X.drop(["HeartDisease"], axis=1, inplace=True)

split_train_X, split_pred_X, split_train_y, split_pred_y = train_test_split(X, y, random_state=1)

categorical_cols = ["Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", "Sex", "AgeCategory", "Race", "Diabetic", "PhysicalActivity", "GenHealth", "Asthma", "KidneyDisease", "SkinCancer"]
numerical_cols = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]
cols = categorical_cols + numerical_cols  # Tror måske ikke det her giver mening, da jeg tager alle columns med anyway.. ?

X_train = split_train_X[cols].copy()  # ved ikke helt hvorfor man skal bruge .copy() her
X_valid = split_pred_X[cols].copy()
X_test = X[cols]

# categorical_transformer = OneHotEncoder()
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[('cat', categorical_transformer, categorical_cols)])

model = LogisticRegression(random_state=0)

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])
# clf.fit(X_train, split_train_y)
# preds = clf.predict(X_valid)

print("X", X_valid.shape)
print("y", split_train_y.shape)
# Et eller andet steder bliver der droppet en masse linjer fra X - må vel ske under preprocessing.. ?

scores = cross_val_score(model, X_valid, split_train_y, cv=5)
print('Cross validation scores', scores)

