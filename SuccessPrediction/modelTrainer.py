# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import cPickle as c

#  Load the dataset
data = pd.read_csv("fbkfcsldata.csv")

#IND=data.as_matrix(columns=['Likes', 'Comments'])
IND=data.as_matrix(columns=['Views'])

#Defining the splits for categories. 20-100 will be poor quality, 100-1000 will be average, 1000-10000 will be great
bins = [20,100,1000,10000]

#0 for low quality, 1 for average, 2 for great quality
quality_labels=[0,1,2]
data['quality_categorical'] = pd.cut(data['Likes'], bins=bins, labels=quality_labels, include_lowest=True)

# Split the data into features and target label
quality_raw = data['quality_categorical']
#features_raw = data
features_raw= data.drop(labels='quality_categorical', axis=1)

# Split the 'features' and 'likes' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_raw,
                                                    quality_raw,
                                                    test_size = 0.2,
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

def train_predict_evaluate(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: quality training set
       - X_test: features testing set
       - y_test: quality testing set
    '''

    results = {}

    """
    Fit/train the learner to the training data using slicing with 'sample_size'
    using .fit(training_features[:], training_labels[:])
    """
    start = time()  # Get start time of training
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])  # Train the model
    end = time()  # Get end time of training

    # Calculate the training time
    results['train_time'] = end - start

    """
    Get the predictions on the first 60 training samples(X_train),
    and also predictions on the test set(X_test) using .predict()
    """
    start = time()  # Get start time
    predictions_train = learner.predict(X_train[:100])
    predictions_test = learner.predict(X_test)

    end = time()  # Get end time

    # Calculate the total prediction time
    results['pred_time'] = end - start

    # Compute accuracy on the first 60 training samples which is y_train[:60]
    results['acc_train'] = accuracy_score(y_train[:100], predictions_train)

    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)

    # Compute F1-score on the the first 60 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:100], predictions_train, beta=0.5, average='micro')

    # Compute F1-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5, average='micro')

    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))

    # Return the results
    return results

# Initialize the classifier
clf = RandomForestClassifier(max_depth=None, random_state=None)

# Create the parameters or base_estimators list you wish to tune, using a dictionary if needed.
# Example: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}

"""
n_estimators: Number of trees in the forest
max_features: The number of features to consider when looking for the best split
max_depth: The maximum depth of the tree
"""
parameters = {'n_estimators': [10, 20, 30], 'max_features':[1,2,3, None], 'max_depth': [5,6,7, None]}


# Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score, beta=0.5, average="micro")

# Perform grid search on the claszsifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)
#print(grid_obj)

# Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

def save(clf, name):
    with open(name, 'wb') as fp:
        c.dump(clf, fp)
    print "saved"

# Get the estimator
best_clf = grid_fit.best_estimator_
save(best_clf, "success-classifier-reactions.mdl")

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.8f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.8f}".format(fbeta_score(y_test, predictions, beta = 0.5, average="micro")))
print("\nOptimized Model\n------")
print(best_clf)
print("\nFinal accuracy score on the testing data: {:.8f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.8f}".format(fbeta_score(y_test, best_predictions, beta = 0.5,  average="micro")))

# fbdata = [[7000,5,2,3,8,0,150,110,0]]
# #fbdata.reshape(-1,1)
# quality= best_clf.predict(fbdata)
# print("Predicted Quality for the post is: ", quality[0])