# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for MP3. You should only modify code
within this file and the last two arguments of line 34 in mp3.py
and if you want-- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
from math import log
from collections import Counter, OrderedDict


class Stopwords:
    def __init__(self):
        self.sw = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                  "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                  'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                  'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
                  'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                  'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                  'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                  'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                  'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                  'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                  'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
                  "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
                  'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
                  "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
                  'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'br']

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter=0.8, pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set

    # Define Bag of Words model into positive and negative
    stop_words = Stopwords().sw
    cnt_pos = Counter()
    cnt_neg = Counter()
    for i in range(len(train_set)):
        for w in train_set[i]:
            if w not in stop_words:
                if train_labels[i]:
                    cnt_pos[w] += 1
                else:
                    cnt_neg[w] += 1

    # Limit vocab size to 5000
    # cnt_pos = OrderedDict(cnt_pos.most_common(5000))
    # cnt_neg = OrderedDict(cnt_neg.most_common(5000))

    # Naive Bayes: P(Class|W_1,...,W_k) is proportional to log(P(Positive)) + (sum of log(P(W_k|Class)) from k = 1 to k = n)
    # P(W_k|Class) = (count(W)+alpha)/(n+alpha(V+1))
    # P(Positive) = 0.8 given

    alpha = smoothing_parameter
    neg_prior = (1-pos_prior)
    pos_n = sum(cnt_pos.values())
    neg_n = sum(cnt_neg.values())
    pos_V = len(cnt_pos)
    neg_V = len(cnt_neg)
    pos_likelihood = {}
    neg_likelihood = {}

    # Calculate the likelihoods from train_set
    for w in cnt_pos:
        count_w = cnt_pos[w]
        pos_likelihood[w] = log((count_w+alpha)/(pos_n+(alpha*(pos_V+1))))
    for w in cnt_neg:
        count_w = cnt_neg[w]
        neg_likelihood[w] = log((count_w+alpha)/(neg_n+(alpha*(neg_V+1))))

    # devel part
    dev_label = []
    for i in range(len(dev_set)):
        pos_posterior = 0
        neg_posterior = 0
        for w in dev_set[i]:
            if w not in stop_words:
                # compute probabilities
                if w in cnt_pos:
                    pos_posterior += pos_likelihood[w]
                # unknown word
                if w not in cnt_pos:
                    pos_posterior += log(alpha / (pos_n + (alpha * (pos_V + 1))))
                if w in cnt_neg:
                    neg_posterior += neg_likelihood[w]
                # unknown word
                if w not in cnt_neg:
                    neg_posterior += log(alpha / (neg_n + (alpha * (neg_V + 1))))
        pos_posterior += log(pos_prior)
        neg_posterior += log(neg_prior)
        if pos_posterior > neg_posterior:
            dev_label.append(1)
        else:
            dev_label.append(0)

    return dev_label

def bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter=0.7, bigram_smoothing_parameter=0.0000000001, bigram_lambda=0.01, pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    unigram_smoothing_parameter - The smoothing parameter for unigram model (same as above) --laplace (1.0 by default)
    bigram_smoothing_parameter - The smoothing parameter for bigram model (1.0 by default)
    bigram_lambda - Determines what fraction of your prediction is from the bigram model and what fraction is from the unigram model. Default is 0.5
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set using a bigram model
    stop_words = Stopwords().sw
    cnt_pos = Counter()
    cnt_neg = Counter()
    for i in range(len(train_set)):
        for j in range(len(train_set[i])-1):
            w = tuple(train_set[i][j:j+2])
            if train_labels[i]:
                cnt_pos[w] += 1
            else:
                cnt_neg[w] += 1

    alpha = bigram_smoothing_parameter
    neg_prior = (1 - pos_prior)
    pos_n = sum(cnt_pos.values())
    neg_n = sum(cnt_neg.values())
    pos_V = len(cnt_pos)
    neg_V = len(cnt_neg)
    pos_likelihood = {}
    neg_likelihood = {}

    # Calculate the likelihoods from train_set
    for w in cnt_pos:
        count_w = cnt_pos[w]
        pos_likelihood[w] = log((count_w + alpha) / (pos_n + (alpha * (pos_V + 1))))
    for w in cnt_neg:
        count_w = cnt_neg[w]
        neg_likelihood[w] = log((count_w + alpha) / (neg_n + (alpha * (neg_V + 1))))

    # devel part
    dev_label = []
    unig_pos_posterior, unig_neg_posterior = unigram_model(train_set, train_labels, dev_set, unigram_smoothing_parameter)
    for i in range(len(dev_set)):
        bigr_pos_posterior = 0
        bigr_neg_posterior = 0
        for j in range(len(dev_set[i])-1):
            w = tuple(dev_set[i][j:j + 2])
            # compute probabilities
            if w in cnt_pos:
                bigr_pos_posterior += pos_likelihood[w]
            # unknown word
            if w not in cnt_pos:
                bigr_pos_posterior += log(alpha / (pos_n + (alpha * (pos_V + 1))))
            if w in cnt_neg:
                bigr_neg_posterior += neg_likelihood[w]
            # unknown word
            if w not in cnt_neg:
                bigr_neg_posterior += log(alpha / (neg_n + (alpha * (neg_V + 1))))
        bigr_pos_posterior += log(pos_prior)
        bigr_neg_posterior += log(neg_prior)
        pos = (1-bigram_lambda)*unig_pos_posterior[i] + bigram_lambda*bigr_pos_posterior
        neg = (1-bigram_lambda)*unig_neg_posterior[i] + bigram_lambda*bigr_neg_posterior
        if pos > neg:
            dev_label.append(1)
        else:
            dev_label.append(0)

    return dev_label

def unigram_model(train_set, train_labels, dev_set, smoothing_parameter, pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set

    # Define Bag of Words model into positive and negative
    stop_words = Stopwords().sw
    cnt_pos = Counter()
    cnt_neg = Counter()
    for i in range(len(train_set)):
        for w in train_set[i]:
            if w not in stop_words:
                if train_labels[i]:
                    cnt_pos[w] += 1
                else:
                    cnt_neg[w] += 1

    # Limit vocab size to 5000
    # cnt_pos = OrderedDict(cnt_pos.most_common(5000))
    # cnt_neg = OrderedDict(cnt_neg.most_common(5000))

    # Naive Bayes: P(Class|W_1,...,W_k) is proportional to log(P(Positive)) + (sum of log(P(W_k|Class)) from k = 1 to k = n)
    # P(W_k|Class) = (count(W)+alpha)/(n+alpha(V+1))
    # P(Positive) = 0.8 given

    alpha = smoothing_parameter
    neg_prior = (1-pos_prior)
    pos_n = sum(cnt_pos.values())
    neg_n = sum(cnt_neg.values())
    pos_V = len(cnt_pos)
    neg_V = len(cnt_neg)
    pos_likelihood = {}
    neg_likelihood = {}

    # Calculate the likelihoods from train_set
    for w in cnt_pos:
        count_w = cnt_pos[w]
        pos_likelihood[w] = log((count_w+alpha)/(pos_n+(alpha*(pos_V+1))))
    for w in cnt_neg:
        count_w = cnt_neg[w]
        neg_likelihood[w] = log((count_w+alpha)/(neg_n+(alpha*(neg_V+1))))

    # devel part
    dev_label = []
    pos_list = []
    neg_list = []
    for i in range(len(dev_set)):
        pos_posterior = 0
        neg_posterior = 0
        for w in dev_set[i]:
            if w not in stop_words:
                # compute probabilities
                if w in cnt_pos:
                    pos_posterior += pos_likelihood[w]
                # unknown word
                if w not in cnt_pos:
                    pos_posterior += log(alpha / (pos_n + (alpha * (pos_V + 1))))
                if w in cnt_neg:
                    neg_posterior += neg_likelihood[w]
                # unknown word
                if w not in cnt_neg:
                    neg_posterior += log(alpha / (neg_n + (alpha * (neg_V + 1))))
        pos_posterior += log(pos_prior)
        neg_posterior += log(neg_prior)
        pos_list.append(pos_posterior)
        neg_list.append(neg_posterior)

    return pos_list, neg_list
