import cPickle as c
import os
import pandas as pd
from collections import Counter
import unicodedata

def load(clf_file):
    with open(clf_file) as fp:
        clf = c.load(fp)
    return clf


def make_dict():
    words = []

    texts = pd.read_csv('spam.csv', encoding='latin-1')
    texts.head()
    texts = texts.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    texts = texts.rename(columns={'v1': 'label', 'v2': 'message'})
    messages = texts['message'].tolist();

    for message in messages:
        message = unicodedata.normalize('NFKD', message).encode('ascii', 'ignore')
        words += message.split(" ")

    for i in range(len(words)):
        if not words[i].isalpha():
            words[i] = ""

    dictionary = Counter(words)
    del dictionary[""]
    return dictionary.most_common(3000)


clf = load("text-classifier.mdl")
d = make_dict()


while True:
    features = []
    inp = raw_input(">").split()
    if inp[0] == "exit":
        break
    for word in d:
        features.append(inp.count(word[0]))
    res = clf.predict([features])
    print ["Not Spam", "Spam!"][res[0]]