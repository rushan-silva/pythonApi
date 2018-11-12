import cPickle as c

def detector(data):

    def load(clf_file):
        with open(clf_file) as fp:
            clf = c.load(fp)
        return clf

    best_clf = load("SuccessPrediction/success-classifier-emotions.mdl")

    # fbdata = [[7, 5, 2, 20, 2, 4, 8]]
    # fbdata.reshape(-1,1)

    fbdata = [data]
    quality = best_clf.predict(fbdata)  # type: int
    print("Predicted Quality for the post is: ", quality[0])
    return str(quality[0])