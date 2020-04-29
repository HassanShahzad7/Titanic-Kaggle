# importing necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

# importing training and test data sets
test = pd.read_csv('./test.csv')
train = pd.read_csv('./train.csv')

# initializing LabelEncoder to encode categorical data
le = LabelEncoder()

# filling all nan entries in test and train to -1
train['Age'] = train['Age'].fillna(-1)
test['Age'] = test['Age'].fillna(-1)
test['Fare'] = test['Fare'].fillna(-1)

# converting categorical data to list
em1 = list(train['Embarked'])
em2 = list(train['Sex'])
em3 = list(test['Embarked'])
em4 = list(test['Sex'])

# label encoding the train data
embarked = le.fit_transform(em1)
sex = le.fit_transform(em2)
zipper1 = list(zip(embarked, sex))

# label encoding the test data
embarked = le.fit_transform(em3)
sex = le.fit_transform(em4)
zipper = list(zip(embarked, sex))

# setting up DataFrames -> emm for train data and emm1 for test data
emm = pd.DataFrame(zipper1, columns = ['Embarked', 'Sex'])
emm1 = pd.DataFrame(zipper, columns = ['Embarked', 'Sex'])

# features which want to be included
features = ['PassengerId', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Age']

# using only required features in both test and train data sets
X = train[features]
X_test = test[features]
X_test = pd.concat([X_test, emm1], axis = 1)
X = pd.concat([X, emm], axis = 1)
y = train['Survived']

# initializing Random Forest
model = RandomForestClassifier(n_estimators= 450, max_depth= 9, max_leaf_nodes= 49, random_state = 123)

# fitting the model
model.fit(X, y)

# predicting the model
pred = model.predict(X_test)

# setting up my predictions as a data frame named tester
pid = test['PassengerId']
zipper = list(zip(pid, pred))
predi = pd.DataFrame(pred, columns = ['Survived'])
tester = pd.DataFrame(pid, columns = ['PassengerId'])
tester = pd.concat([tester, predi], axis = 1)
saver = tester.to_csv('tester.csv')
