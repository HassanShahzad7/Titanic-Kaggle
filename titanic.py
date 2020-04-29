from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

test = pd.read_csv('./test.csv')
train = pd.read_csv('./train.csv')

le = LabelEncoder()
train['Age'] = train['Age'].fillna(-1)
test['Age'] = test['Age'].fillna(-1)
test['Fare'] = test['Fare'].fillna(-1)
em1 = list(train['Embarked'])
em2 = list(train['Sex'])
embarked = le.fit_transform(em1)
sex = le.fit_transform(em2)
zipper1 = list(zip(embarked, sex))
em3 = list(test['Embarked'])
em4 = list(test['Sex'])
embarked = le.fit_transform(em3)
sex = le.fit_transform(em4)
zipper = list(zip(embarked, sex))
emm = pd.DataFrame(zipper1, columns = ['Embarked', 'Sex'])
emm1 = pd.DataFrame(zipper, columns = ['Embarked', 'Sex'])

features = ['PassengerId', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Age']
X = train[features]
X_test = test[features]
X_test = pd.concat([X_test, emm1], axis = 1)
X = pd.concat([X, emm], axis = 1)
y = train['Survived']

model = RandomForestClassifier(n_estimators= 450, max_depth= 9, max_leaf_nodes= 49, random_state = 123)
model.fit(X, y)
pred = model.predict(X_test)
pid = test['PassengerId']
zipper = list(zip(pid, pred))
predi = pd.DataFrame(pred, columns = ['Survived'])
tester = pd.DataFrame(pid, columns = ['PassengerId'])
tester = pd.concat([tester, predi], axis = 1)
saver = X_test.to_csv('tester.csv')
