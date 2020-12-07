import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ref: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

# a 2d list of training sets, where each index should contains array of 3 elements
train_tweet_id_list = []
train_text_list = []
train_q1_label_list = []
# read train file with utf-8
train_file = open("data/covid_training.tsv", "r", encoding="utf-8")
train_list = train_file.readlines()
train_file.close()
# remove headear row
del train_list[0]
# put every training set with tweet_id, text, and q1_label into 2d array
for row in train_list:
    train_string_list = str(row).split("\t")
    train_tweet_id_list.append(train_string_list[0])
    train_text_list.append(train_string_list[1].lower())
    train_q1_label_list.append(train_string_list[2])
# Tokenizing text with scikit-learn
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_text_list)
# From occurrences to frequencies
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# read test file with utf-8
test_file = open("data/covid_test_public.tsv", "r", encoding="utf-8")
test_list = test_file.readlines()
test_file.close()
# a 2d list of testing sets, where each index should contains array of 3 elements
test_tweet_id_list = []
test_text_list = []
test_q1_label_list = []
# put every testing set with tweet_id, text, and q1_label into 2d array
for row in test_list:
    test_string_list = str(row).split("\t")
    test_tweet_id_list.append(test_string_list[0])
    test_text_list.append(test_string_list[1].lower())
    test_q1_label_list.append(test_string_list[2])

# Training a classifier
clf = MultinomialNB(alpha=0.01).fit(X_train_tfidf, train_q1_label_list)

X_new_counts = count_vect.transform(test_text_list)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

# Output
output_trace_file = open("output/trace_NB-BOW-OV.txt", "w")
# TODO 2.2.2 3. per-class recall (yes-R, no-R)
for id, predict, result in zip(test_tweet_id_list, predicted, test_q1_label_list):
    output_trace_file.writelines(id + "  " + predict + "  " + score? + "  " + ("correct" if (predict == result) else "wrong") + "\r")
output_trace_file.close()

# https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
# TODO fix precision_score, recall_score, f1_score
output_eval_file = open("output/eval_NB-BOW-OV.txt", "w")
output_eval_file.write(
    metrics.accuracy_score(test_q1_label_list, predicted.tolist()) + "\r" +
    metrics.precision_score(test_q1_label_list, predicted.tolist()) + "\r" +
    metrics.recall_score(test_q1_label_list, predicted.tolist()) + "\r" +
    metrics.f1_score(test_q1_label_list, predicted.tolist()) + "\r"
)
output_eval_file.close()