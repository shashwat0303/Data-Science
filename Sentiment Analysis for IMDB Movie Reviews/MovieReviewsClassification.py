from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

posRev = open('PositiveReviews.txt', 'r').read()
negRev = open('NegativeReviews.txt', 'r').read()

posRev = posRev.split('\t\t\t')
negRev = negRev.split('\t\t\t')

vectorizer = CountVectorizer(min_df=0)

def make_xy(posRev, negRev, vectorizer):
    reviews = posRev
    reviewType = np.ones(len(posRev)).tolist()
    for rev in negRev:
        reviews.append(rev)
        reviewType.append(0)

    vectorizer.fit(reviews)

    X = vectorizer.transform(reviews)
    X = X.toarray()
    y = reviewType
    y = np.array(y)

    return X, y

X, y = make_xy(posRev, negRev, vectorizer)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
clf = MultinomialNB()
clf.fit(Xtrain, ytrain)

trainAcc = clf.score(Xtrain, ytrain)
testAcc = clf.score(Xtest, ytest)

print "Training Accuracy for the given set of reviews = ", trainAcc
print "Testing Accuracy for the given set of reviews = ", testAcc


def calibration_plot(clf, Xtest, ytest):
    prob = clf.predict_proba(X)[:, 1]
    output = y
    data = pd.DataFrame(dict(prob = prob, outcome = output))

    bins = np.linspace(0, 1, 20)
    cuts = pd.cut(prob, bins)

    calibrated = data.groupby(cuts).outcome.agg(['mean', 'count'])
    calibrated['avgProbability'] = (bins[1:] + bins[:-1]) / 2
    calibrated['sigma'] = np.sqrt(calibrated.avgProbability * (1 - calibrated.avgProbability) / calibrated['count'])
    plt.errorbar(calibrated.avgProbability, calibrated['mean'], calibrated.sigma)
    plt.show()

def log_likelihood(clf, X, y):
    prob = clf.predict_log_proba(X)
    rotten = y == 0
    fresh = y == 1
    l = prob[rotten, 0].sum() + prob[fresh, 1].sum()
    return l

def cv_score(clf, X, y, score_func):
    result = 0
    nfold = 5
    kf = KFold(n_splits = nfold, shuffle = True, random_state = None)
    for train, test in kf.split(X):
        clf.fit(X[train], y[train])
        result = result + score_func(clf, X[test], y[test])
    return result / nfold

'''alphas = [0, .1, 1, 5]
min_dfs = [1e-3, 1e-2, 1e-1]

#Find the best value for alpha and min_df, and the best classifier
best_alpha = None
best_min_df = None
max_loglike = -np.inf

for alpha in alphas:
    for min_df in min_dfs:
        vectorizer = CountVectorizer(min_df = min_df)
        X, y = make_xy(posRev, negRev, vectorizer)
        clf = MultinomialNB(alpha = alpha)
        loglike = cv_score(clf, X, y, log_likelihood)
        if (loglike > max_loglike):
            max_loglike = loglike
            best_min_df = min_df
            best_alpha = alpha
        else:
            continue

print "alpha: %f" % best_alpha
print "min_df: %f" % best_min_df


vectorizer = CountVectorizer(min_df=best_min_df)
X, y = make_xy(posRev, negRev, vectorizer)
xtrain, xtest, ytrain, ytest = train_test_split(X, y)

clf = MultinomialNB(alpha=best_alpha).fit(xtrain, ytrain)

calibration_plot(clf, xtest, ytest)

# Your code here. Print the accuracy on the test and training dataset
training_accuracy = clf.score(xtrain, ytrain)
test_accuracy = clf.score(xtest, ytest)

print "Accuracy on training data: %0.2f" % (training_accuracy)
print "Accuracy on test data:     %0.2f" % (test_accuracy)'''


words = np.array(vectorizer.get_feature_names())

X = np.eye(Xtest.shape[1])
probs = clf.predict_log_proba(X)[:, 0]
ind = np.argsort(probs)

good_words = words[ind[:10]]
bad_words = words[ind[-10:]]

good_prob = probs[ind[:10]]
bad_prob = probs[ind[-10:]]

print "Good words\t     P(fresh | word)"
for w, p in zip(good_words, good_prob):
    print "%20s" % w, "%0.2f" % (1 - np.exp(p))

print "Bad words\t     P(fresh | word)"
for w, p in zip(bad_words, bad_prob):
    print "%20s" % w, "%0.2f" % (1 - np.exp(p))

testReview = '''I have just given a 10 for Thieves Highway, I mention this for two reasons one to prove I'm not a git who only gives bad reviews but 2 because the theme of the film has the same thread namely the falling in love with a woman of the night.

We all know pretty Woman is a chick flick but you can't avoid them all, they'll eventually get you. Pretty Woman for me does two things, two terrible horrible ghastly things, firstly it portrays prostitution as a career more akin to that of a dancer, you know with absolutely great friends, leg warmers lots of giggling, borrowing each others make up. You see in the reality of Pretty Woman the prostitute and this is a street walker Prostitute we're talking about here, has a great life, she's healthy happy with only the occasional whimper to explain her predicament. My feeling is this 'happy Hooker' type protagonist is a lot more palatable than an even nearly realistic character, which for me begs the question if you make a movie about a type of person but are too chicken scared to adorn that player with the characteristics familiar to that role then why do it? If I make a film about a chef but don't want him to cook or talk about food or wear a white hat then why make a film about a chef in the first place? By bailing out and turning the hooker into a respectable dancer type the story misses the point completely and consequently never indulges in any of the moral or social questions that it could have, what a cop out, really really lame.

Secondly, 'Pretty Woman' insults romance itself, Edward Lewis played by Richard Gere has no clue how to seduce or romance this 'lady' that is without his plastic friend, yep don't leave home without it, especially if you are a moron in a suit who has no imagination. 8 out of 10 of his romantic moments involve splashing cash in one way or another, even when he first meets her it's the Lotus Esprit turbo that does all the work, necklaces here diamonds there limos over there, money money money, where's the charm? where's the charisma, don't mention that attempt at the piano please.

Girls who like this film will also be girls who like shopping more than most. Guys who like this film will not even have realized that old Eddy has less charm than a calculator, as they probably don't either so it wont have registered. More importantly anyone who likes this film will hate 'Thieves Highway' a wonderful story of which part is based on the same subject.

I'll finish on a song:

Pretty woman hangin round the street Pretty woman, the kind I like to treat Pretty woman, I don't believe you You're not the truth No one could spend as much as you Mercy

Pretty woman, wont you pardon me Pretty woman, I couldn't help but see Pretty woman, and you look lovely as can be do you lack imagination just like me

Pretty woman, shop a while Pretty woman, talk a while Pretty woman, sell your smile to me Pretty woman, yeah, yeah, yeah Pretty woman, look my way Pretty woman, say you'll stay with me..and I'll pay you..I'll treat you right'''

tr = vectorizer.transform([testReview])
print clf.predict_proba(tr)
