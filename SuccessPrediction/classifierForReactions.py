import cPickle as c

def classifier(dataArr):

    def load(clf_file):
        with open(clf_file) as fp:
            clf = c.load(fp)
        return clf

    best_clf = load("SuccessPrediction/success-classifier-reactions.mdl")

    # fbdata = [[7000,5,2,3,8,0,150,110,0]]
    #fbdata.reshape(-1,1)

    fbdata = [dataArr]
    quality = best_clf.predict(fbdata)
    print("Predicted Quality for the post is: ", quality[0])
    return str(quality[0])