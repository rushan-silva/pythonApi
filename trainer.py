import os
from collections import Counter
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import cPickle as c
from IPython.display import display
import unicodedata

def save(clf, name):
    with open(name, 'wb') as fp:
        c.dump(clf, fp)
    print "saved"

def make_dict():
    words = []

    texts = pd.read_csv('spam.csv', encoding='latin-1')
    texts.head()
    texts = texts.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    texts = texts.rename(columns={'v1': 'label', 'v2': 'message'})
    messages = texts['message'].tolist();

    c = len(messages)
    for message in messages:
        print message
        message = unicodedata.normalize('NFKD', message).encode('ascii', 'ignore')
        words += message.split(" ")
        print c
        c -= 1

    print words

    for i in range(len(words)):
        if not words[i].isalpha():
            words[i] = ""

    dictionary = Counter(words)
    del dictionary[""]
    print dictionary.most_common(3000)
    return dictionary.most_common(3000)

def make_dataset(dictionary):
    texts = pd.read_csv('spam.csv', encoding='latin-1')
    texts.head()
    texts = texts.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    texts = texts.rename(columns={'v1': 'label', 'v2': 'message'})
    messages = texts['message'].tolist();
    labelsList = texts['label'].tolist();

    feature_set = []
    labels = []
    c = len(messages)

    for ind, message in enumerate(messages):
        data = []
        words = message.split(' ')
        for entry in dictionary:
            data.append(words.count(entry[0]))
        feature_set.append(data)

        if labelsList[ind] == 'ham':
            labels.append(0)
        if labelsList[ind] == 'spam':
            labels.append(1)

        print c
        c = c - 1
    print labels
    return feature_set, labels

d = make_dict()
features, labels = make_dataset(d)

x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2)

clf = MultinomialNB()
clf.fit(x_train, y_train)

preds = clf.predict(x_test)
print accuracy_score(y_test, preds)
save(clf, "text-classifier.mdl")