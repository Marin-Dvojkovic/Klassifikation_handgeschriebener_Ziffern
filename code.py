# if you execute this it will probably take a while since the dataset is pretty big

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV

#import the dataset with handwritten digits
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)  # feature
assert y_train.shape == (60000,)  # target
# dataset contains 60.000 images

# reshape features array(x_train) to the 2d array with n rows (n= 60000) and m columns
# m = number of pixels in the image 728
x, y, z = x_train.shape
x_train = x_train.reshape(x, y * z)

# classes are the unique values in y_train
# so let's first eliminate the duplicates from it
classes = np.unique(y_train)

# print some info about the feature and target
# print('Classes:', classes)
# print("Feature's shape:", x_train.shape)
# print("Target's shape:", y_train.shape)
# print(f'min: {np.min(x_train)}, max: {np.max(x_train)}')

#x_train, x_test, y_train, y_test = train_test_split(x_train[:6000], y_train[:6000], test_size=0.3, random_state=40)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=40)

# print('x_train.shape:', x_train.shape)
# print('x_test.shape', x_test.shape)
# print('y_train.shape:', y_train.shape)
# print('y_test.shape', y_test.shape)
# print('Proportion of samples per class in train set:')
y = pd.DataFrame(y_train)
# print(y.value_counts(normalize=True))

modelnames = ['KNeighborsClassifier', 'DecisionTreeClassifier', 'LogisticRegression', 'RandomForestClassifier']
scores = []
models = [KNeighborsClassifier(), DecisionTreeClassifier(random_state=40),
          LogisticRegression(random_state=40, solver="liblinear"),
          RandomForestClassifier(random_state=40)]


# fit the model make the prediction and print the accuracy
def fit_predict_eval(model, features_train, features_test, target_train, target_test, arr):
    # fit the model
    model = model.fit(features_train, target_train)
    # make a prediction
    y_pred = model.predict(features_test)
    # calculate accuracy and save it to score
    score = accuracy_score(target_test, y_pred)
    arr += [score]
    print(f'Model: {model}\nAccuracy: {score}\n')


# execute fit_predict val for all the models
# for i in models:
#    fit_predict_eval(i, x_train, x_test, y_train, y_test, scores)

# get maximum score and round, so it looks super beautiful
# j = scores.index(max(scores))
# k = round(max(scores), 3)

# question: which model has the highest accuracy and what is the accuracy?
# print(f'The answer to the question: {modelnames[j]} - {k}')

n = Normalizer()

x_train_norm = n.transform(x_train)
x_test_norm = n.transform(x_test)

scores1 = []
# for i in models:
#    fit_predict_eval(i, x_train_norm, x_test_norm, y_train, y_test, scores1)

# 1st question: Does the normalization have a positive impact in general?
# print('The answer to the 1st question: yes')

# 2nd question: Which two models show the best scores?
# j = scores1.index(max(scores1))
# k = round(scores1[j], 3)
# scores1[j] = 0
# j1 = scores1.index(max(scores1))
# k1 = round(scores1[j1], 3)
# print(f'The answer to the 2nd question: {modelnames[j]} - {k}, {modelnames[j1]} - {k1} ')

# after experimenting to find out which model ist the best,
# i now tune the hyperparameters.
# I will choose the normalized data representation since it performed better in previous stages
# K-nearest neighbours
knn = GridSearchCV(estimator=KNeighborsClassifier(),
                   param_grid=dict(n_neighbors = [3, 4], weights = ['uniform', 'distance'], algorithm = ['auto', 'brute']),
                   scoring='accuracy', n_jobs=-1)
knn.fit(x_train_norm, y_train)
print('K-nearest neighbours algorithm')
print('best estimator: ', knn.best_estimator_)
optimalKnn = knn.best_estimator_
optimalKnn.fit(x_train_norm, y_train)
predk = optimalKnn.predict(x_test)
print('accuracy: ', accuracy_score(predk, y_test))
print()

# Random Forest
rf = GridSearchCV(estimator=RandomForestClassifier(random_state=40),
                  param_grid=dict(n_estimators = [300, 500], max_features = ['auto', 'log2'], class_weight = ['balanced', 'balanced_subsample']), scoring='accuracy', n_jobs=-1)
rf.fit(x_train_norm, y_train)
print('Random forest algorithm')
print('best estimator: ', rf.best_estimator_)
optimalrf = rf.best_estimator_
optimalrf.fit(x_train_norm, y_train)
predr = optimalrf.predict(x_test_norm)
print('accuracy: ', accuracy_score(predr, y_test))
