# %%
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# %%
def word_tokenization(text):
    vectorizer = CountVectorizer()
    
    # feature frequency
    tokenized_text = (vectorizer.fit_transform(text)).toarray()
    # feature name
    tokenized_names = vectorizer.get_feature_names()
    
    row, column = tokenized_text.shape
    features_text = np.zeros((row, column), dtype=object)
    for r in range(row):
        for c in range(column):
            # tuple containing feature name and frequency
            features_text[r][c] = (tokenized_names[c], tokenized_text[r][c])
    return features_text

# %%
def prior_probabilities(q1_label):
    number_of_yes, number_of_no = 0, 0
    
    # per class prior probabilities
    for i in range(len(q1_label)):
        # probability that the class is "yes"
        if (q1_label[i] == "yes"):
            number_of_yes = number_of_yes + 1
        # probability that the class is "no"
        else:
            number_of_no = number_of_no + 1
    
    probability_of_yes, probability_of_no = (number_of_yes/len(q1_label)), (number_of_no/len(q1_label)) 
    return probability_of_yes, probability_of_no

# %%
def conditional_probabilities(text, q1_label, SMOOTHING):
    row = len(set(q1_label))
    column = text.shape[1]
    probabilities = np.zeros((row, column), dtype=object)
    smoothing_vocabulary = SMOOTHING * column
    
    # total number of words in each class
    total_row, total_column = text.shape
    total_words_in_yes, total_words_in_no = 0, 0
    for r in range(total_row):
        # total number of words in yes
        if (q1_label[r] == "yes"):
            for c in range(total_column):
                total_words_in_yes = total_words_in_yes + text[r][c][1]
        # total number of words in no
        else: 
            for c in range(total_column):
                total_words_in_no = total_words_in_no + text[r][c][1]
    
    # frequency of word per class
    for c in range(column):
        number_of_word_in_yes, number_of_word_in_no = 0, 0
        for r in range(row):
            # frequency of word in yes
            if (q1_label[r] == "yes"):
                number_of_word_in_yes = number_of_word_in_yes + text[r][c][1]
            # frequency of word in no
            else:
                number_of_word_in_no = number_of_word_in_no + text[r][c][1]
        
        #row 1 = yes
        probability_with_class_yes = (number_of_word_in_yes + SMOOTHING)/(total_words_in_yes + smoothing_vocabulary)
        probabilities[0][c] = (text[0][c][0], probability_with_class_yes)
        #row 2 = no
        probability_with_class_no = (number_of_word_in_no + SMOOTHING)/(total_words_in_no + smoothing_vocabulary)
        probabilities[1][c] = (text[0][c][0], probability_with_class_no)
    
    return probabilities

# %%
def trim_vocabulary(conditionals, text):
    matches = []
    
    text_column = text.shape[1]
    conditional_column = conditionals.shape[1]
    
    for tc in range(text_column):
        for cc in range(conditional_column):
            if (text[0][tc][0] == conditionals[0][cc][0]):
                matches.append((tc, cc))
    
    return matches

# %%
def class_prediction(yes, no, conditionals, text, matches): 
    text_row = text.shape[0]
    prediction = []
    
    for tr in range(text_row):
        score_yes, score_no = math.log(yes), math.log(no)
        for i in range(len(matches)):
            text_index = matches[i][0]
            conditional_index = matches[i][1]
            if (text[tr][text_index][1] > 0):
                score_yes = score_yes + math.log(conditionals[0][conditional_index][1])
                score_no = score_no + math.log(conditionals[1][conditional_index][1])
        if (score_yes > score_no):
            prediction.append("yes")
        else:
            prediction.append("no")
    
    return prediction

# %%
def data_processing(training_file, testing_file, SMOOTHING):
    # training: hypothesis and evidence
    train_dataset = (pd.read_csv(training_file, sep='\t')).to_numpy()
    train_tweet_id = train_dataset[:,0]
    train_text = word_tokenization(train_dataset[:,1])
    train_q1_label = train_dataset[:,2]
    
    # prior probabilities
    yes, no = prior_probabilities(train_q1_label)
    
    # conditional probabilities 
    conditionals = conditional_probabilities(train_text, train_q1_label, SMOOTHING)
    
    # testing: hypothesis and evidence
    test_dataset = (pd.read_csv(testing_file, sep='\t')).to_numpy()
    test_tweet_id = test_dataset[:,0]
    test_text = word_tokenization(test_dataset[:,1])
    test_q1_label = test_dataset[:,2]
    
    
    # return index of matching words
    matches = trim_vocabulary(conditionals, test_text)
    
    # prediction
    prediction = class_prediction(yes, no, conditionals, test_text, matches)
    
    print(prediction)

# %%
# processing datasets
training = '../data/covid_training.tsv'
testing = '../data/covid_test_public.tsv'
data_processing(training, testing, 0.01)

# %%
