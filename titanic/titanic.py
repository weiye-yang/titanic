import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import metrics
from sklearn import ensemble
from sklearn import model_selection
import matplotlib.pyplot as plt
import seaborn as sns
import tools

#read in data
train_all = pd.read_csv("train.csv")
test_all = pd.read_csv("test.csv")
#combine all data to clean together
all_data = train_all.copy()
del all_data["Survived"]
all_data = all_data.append(test_all).reset_index(drop = True)

#impute gaps of Embarked with most common value
all_data["Embarked"] = all_data["Embarked"].fillna(tools.mostCommon(all_data["Embarked"].values[np.logical_not(pd.isnull(all_data["Embarked"]))]))
#add prefix to name, i.e. C becomes Em_C
all_data["Embarked"] = "Em_" + all_data["Embarked"]
#add dummy variables for Embarked
dummies = pd.get_dummies(all_data["Embarked"])
all_data = pd.concat([all_data, dummies], 1)

#replace existing cabin data with 1
all_data["Cabin"].values[pd.notnull(all_data["Cabin"])] = 1
#fill in missing Cabin data with 0. whether or not this information exists seems a good indicator of survival
all_data["Cabin"].values[pd.isnull(all_data["Cabin"])] = 0

#for unrecorded ages, impute the median of all other passengers that have the same Pclass, Sex and Embarked
for i in range(0,len(all_data)):
    if pd.isnull(all_data["Age"][i]):
        comparables = np.logical_and( np.logical_and(all_data["Pclass"].values == all_data["Pclass"][i],all_data["Sex"].values == all_data["Sex"][i]),all_data["Embarked"].values == all_data["Embarked"][i])
        all_data["Age"].values[i] = np.median(tools.delNan(all_data["Age"].values[comparables]))

#for unrecorded fares, impute the median of all other passengers that have the same Pclass, Sex and Embarked
for i in range(0,len(all_data)):
    if pd.isnull(all_data["Fare"][i]):
        comparables = np.logical_and( np.logical_and(all_data["Pclass"].values == all_data["Pclass"][i],all_data["Sex"].values == all_data["Sex"][i]),all_data["Embarked"].values == all_data["Embarked"][i])
        all_data["Fare"].values[i] = np.median(tools.delNan(all_data["Fare"].values[comparables]))

#we no longer need the Embarked or Cabin field
del all_data["Embarked"]

#turn values of Sex into numbers for convenience
all_data["Sex"].values[all_data["Sex"] == "male"] = 0
all_data["Sex"].values[all_data["Sex"] == "female"] = 1

#isolate titles and clean. Differentiate between Generic, Rev., Nobility and Military
all_data.insert(3,"Title","")
for i in range(0,len(all_data)):
    firstname = all_data["Name"][i].split(",")[1].strip()
    title = firstname.split(" ")[0]
    if title == "Capt." or title == "Col." or title == "Major.":
        all_data["Title"].values[i] = "Mil"
    elif title == "Jonkheer." or title == "Lady." or title == "Sir." or title == "the":
        all_data["Title"].values[i] = "Nob"
    elif title=="Dona." or title=="Miss." or title=="Mlle." or title=="Mme." or title=="Mrs." or title=="Don." or title=="Master." or title=="Ms." or title=="Mr." or title== "Dr.":
        all_data["Title"].values[i] = "Gen"
    else:
        all_data["Title"].values[i] = title
del all_data["Name"]
#add dummy variables for title
dummies = pd.get_dummies(all_data["Title"])
all_data = pd.concat([all_data, dummies], 1)
del all_data["Title"]

#add family size feature
all_data.insert(6,"FamSize",all_data["SibSp"].values + all_data["Parch"].values + 1)

#remove columns which we do not consider useful
del all_data["Ticket"]

#we're done with modifying the data so split it back up into training and test sets, removing feature PassengerId
#as it will definitely not help us with fitting!
del all_data["PassengerId"]
X_train_all = all_data[:len(train_all)]
y_train_all = train_all["Survived"]
X_test_all = all_data[len(train_all):]

'''
Discussion of various classification methods:
K-nearest neighbours: could be tractable, but is locally sensitive to data
Naive Bayes: assumes conditional independence of features given the value of the dependent variable. This isn't true.
Logistic regression: this would assume that the odds of survival are dependent in a linear fashion on the explanatory data, which is not a good assumption.
Support vector machines require either that the data can be classified on either side of a hyperplane, or else need knowledge of a good kernel for the kernel trick. We don't have such information.
'''

#implement a K-fold cross-validation function
def run_kfold(clf, n_folds):
    kf = model_selection.KFold(n_splits = n_folds)
    outcomes = []
    fold = 0
    for train_index, test_index in kf.split(X_train_all):
        fold += 1
        X_train, X_test = X_train_all.values[train_index], X_train_all.values[test_index]
        y_train, y_test = y_train_all.values[train_index], y_train_all.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))     
    mean_outcome = np.mean(outcomes)
    return mean_outcome

#We try Gradient Boosting Classifier
clf = ensemble.GradientBoostingClassifier()
clf.fit(X_train_all,y_train_all)

#using feature_importances_ we delete features that do very little, e.g. importance < 0.015. This should reduce variance
for i in reversed(range(0,len(all_data.columns))):
    if clf.feature_importances_[i] < 0.015:
        del all_data[all_data.columns[i]]

#split datasets again and fit
X_train_all = all_data[:len(train_all)]
X_test_all = all_data[len(train_all):]
clf = ensemble.GradientBoostingClassifier()
clf.fit(X_train_all,y_train_all)

#make prediction!
prediction = clf.predict(X_test_all)
#Finally, output file in the required format
output = pd.concat([test_all["PassengerId"], pd.DataFrame(prediction, columns = ["Survived"])],1)
#output.to_csv("prediction.csv", index = False)