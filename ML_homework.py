import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.ensemble import AdaBoostRegressor,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import zero_one_loss

# load data
train_titanic = pd.read_csv('train_titanic.csv')
test_titanic = pd.read_csv('test_titanic.csv')
breast_data = pd.read_csv("breast_data.csv")
hand_digits = load_digits()
digits_data = hand_digits.data
data_boston = load_boston()

#clean data
features_se= list(breast_data.columns[12:22])
n = 200
train_titanic['Age'].fillna(train_titanic['Age'].mean(), inplace=True)
features_mean= list(breast_data.columns[2:12])
features_worst=list(breast_data.columns[22:32])
test_titanic['Age'].fillna(test_titanic['Age'].mean(), inplace=True)
breast_data.drop("id",axis=1,inplace=True)
breast_data['diagnosis']=breast_data['diagnosis'].map({'M':1,'B':0})
boston_train_x, boston_test_x, boston_train_y, boston_test_y = train_test_split(data_boston.data, data_boston.target, test_size=0.25, random_state=30)
Boosting = AdaBoostRegressor()
train_titanic['Fare'].fillna(train_titanic['Fare'].mean(), inplace=True)
n_iterator = 200
train_titanic['Embarked'].fillna('S', inplace=True)
test_titanic['Embarked'].fillna('S',inplace=True)
breast_feature_chosen = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean']
pd.set_option('display.max_columns', None)
train_breast, test_breast = train_test_split(breast_data, test_size = 0.25)
train_breast_X = train_breast[breast_feature_chosen]
test_titanic['Fare'].fillna(test_titanic['Fare'].mean(),inplace=True)
train_breast_y = train_breast['diagnosis']
sns.countplot(breast_data['diagnosis'],label="Count")
X,y=datasets.make_hastie_10_2(n_samples=12000,random_state=1)
plt.show()
corr = breast_data[features_mean].corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr, annot=True)
plt.show()

test_breast_X = test_breast[breast_feature_chosen]
test_breast_y = test_breast['diagnosis']
scaler = StandardScaler()
train_breast_X = scaler.fit_transform(train_breast_X)
test_breast_X = scaler.transform(test_breast_X)
digits_train_x, digits_test_x, digits_train_y, digits_test_y = train_test_split(digits_data, hand_digits.target, test_size=0.25, random_state=30)
Z_scaler = StandardScaler()
Boosting.fit(boston_train_x,boston_train_y)
Boston_prediction = Boosting.predict(boston_test_x)
train_ss_x = Z_scaler.fit_transform(digits_train_x)
test_ss_x = Z_scaler.transform(digits_test_x)
Boston_MSE = mean_squared_error(boston_test_y, Boston_prediction)
#build model and model comparision between different models
X,y=datasets.make_hastie_10_2(n_samples=12000,random_state=1)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
titanic_train_features = train_titanic[features]
boston_train_x, boston_train_y = X[2000:],y[2000:]
boston_test_x, boston_test_y = X[:2000],y[:2000]
titanic_test_features = test_titanic[features]
KNN = KNeighborsClassifier()
Weak_classifier = DecisionTreeClassifier(max_depth=1,min_samples_leaf=1)
Weak_classifier.fit(boston_train_x, boston_train_y)
Weak_classifier_error = 1.0-Weak_classifier.score(boston_test_x, boston_test_y)
titanic_train_labels = train_titanic['Survived']
Decision_tree_classifier =  DecisionTreeClassifier()
Decision_tree_classifier.fit(boston_train_x, boston_train_y)
Decision_tree_classifier_error = 1.0-Decision_tree_classifier.score(boston_test_x, boston_test_y)
KNN.fit(train_ss_x, digits_train_y)
predict_y = KNN.predict(test_ss_x)
Boosting_classifier = AdaBoostClassifier(base_estimator=Weak_classifier,n_estimators=n)
Boosting_classifier.fit(boston_train_x, boston_train_y)
NNW = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(9, 2), random_state=1)
NNW.fit(train_ss_x, digits_train_y)
predict_y_digits_NNW = NNW.predict(test_ss_x)
dvec_titanic=DictVectorizer(sparse=False)
titanic_train_features=dvec_titanic.fit_transform(titanic_train_features.to_dict(orient='record'))
Dtree = DecisionTreeClassifier(criterion='entropy')
Dtree.fit(titanic_train_features, titanic_train_labels)
breast_model = svm.SVC()
breast_model.fit(train_breast_X,train_breast_y)
prediction=breast_model.predict(test_breast_X)
print(f'The SVM Classifier accuracy score is: {metrics.accuracy_score(prediction,test_breast_y)}')
titanic_test_features=dvec_titanic.transform(titanic_test_features.to_dict(orient='record'))
predicted_labels_titanic = Dtree.predict(titanic_test_features)
decision_tree_accuracy_score = round(Dtree.score(titanic_train_features,titanic_train_labels), 6)
print(f'Decision Tree accuracy score for titanic is {decision_tree_accuracy_score}')
print(f"KNN accuracy score for digits data is {accuracy_score(digits_test_y, predict_y)}" )
print(f'Networking accuracy score for digits data is {accuracy_score(digits_test_y, predict_y_digits_NNW)}')
print("house pricing prediction ", Boston_prediction)
print(f"MSE score in statistic is {round(Boston_MSE,2)} ")
fig_model_comparison = plt.figure()
ax = fig_model_comparison.add_subplot(111)
ax.plot([1,n],[Decision_tree_classifier_error]*2,'k--', label=u'decision tree classifier error rate')
ax.plot([1,n],[Weak_classifier_error]*2, 'k-', label=u'weak classifier error rate')
Boosting_error = np.zeros((n,))
for i,Boston_prediction in enumerate(Boosting_classifier.staged_predict(boston_test_x)):
    Boosting_error[i]=zero_one_loss(Boston_prediction, boston_test_y)
ax.plot(np.arange(n)+1, Boosting_error, label='AdaBoost Test error rate', color='blue')
ax.set_xlabel('itertion number')
ax.set_ylabel('error rate')
leg=ax.legend(loc='upper right',fancybox=True)
plt.show()