import cPickle as c
import os
from collections import Counter
import json
import unicodedata

def detector(commentArray):

    print (commentArray)

    def load(clf_file):
        with open(clf_file) as fp:
            clf = c.load(fp)
        return clf


    def make_dict():
        direc = "texts/"
        files = os.listdir(direc)
        emails = [direc + email for email in files]
        words = []
        c = len(emails)

        for email in emails:
            f = open(email)
            blob = f.read()
            words += blob.split(" ")
            print c
            c -= 1

        for i in range(len(words)):
            if not words[i].isalpha():
                words[i] = ""

        dictionary = Counter(words)
        del dictionary[""]
        return dictionary.most_common(5000)


    clf = load("Detect/text-classifier.mdl")
    d = make_dict()

    for ind, commentObj in enumerate(commentArray):
        print commentObj
        features = []
        comment = commentObj['comment']
        print comment
        # comment = unicodedata.normalize('NFKD', comment).encode('ascii', 'ignore')
        inp = comment.split()
        print inp
        for word in d:
            features.append(inp.count(word[0]))
        res = clf.predict([features])
        print ["Not Spam", "Spam!"][res[0]]
        if (res[0] == 1):
            del commentArray[ind]

    # data = {}
    # data['data'] = commentArray
    # json_data = json.dumps(data)

    print commentArray
    return commentArray
