import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix

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

def conditional_probabilities(text, q1_label, SMOOTHING):
    row = len(set(q1_label))
    column = text.shape[1]
    frequencies = np.zeros((row, column), dtype=int)
    probabilities = np.zeros((row, column), dtype=object)
    smoothing_vocabulary = SMOOTHING * column
    
    total_row, total_column = text.shape
    total_yes, total_no = 0, 0
    for r in range(total_row): 
        for c in range(total_column):
            # "yes" class
            if (q1_label[r] == "yes"):
                total_yes = total_yes + text[r][c][1]
                frequencies[0][c] = frequencies[0][c] + text[r][c][1]
            # "no" class
            else:
                total_no = total_no + text[r][c][1]
                frequencies[1][c] = frequencies[1][c] + text[r][c][1]
                
    for r in range(total_row): 
        for c in range(total_column):
            # "yes" class
            probability = (frequencies[0][c] + SMOOTHING)/(total_yes + smoothing_vocabulary)
            probabilities[0][c] = (text[r][c][0], probability)
            # "no" class
            probability = (frequencies[1][c] + SMOOTHING)/(total_no + smoothing_vocabulary)
            probabilities[1][c] = (text[r][c][0], probability)

    return probabilities

def trim_vocabulary(conditionals, text):
    matches = []
    
    text_column = text.shape[1]
    conditional_column = conditionals.shape[1]
    
    for tc in range(text_column):
        for cc in range(conditional_column):
            if (text[0][tc][0] == conditionals[0][cc][0]):
                matches.append((tc, cc))
    
    return matches

def class_prediction(yes, no, conditionals, text, matches): 
    text_row = text.shape[0]
    prediction = []
    scores = []
    
    for tr in range(text_row):
        score_yes, score_no = math.log10(yes), math.log10(no)
        for i in range(len(matches)):
            text_index = matches[i][0]
            conditional_index = matches[i][1]
            if (text[tr][text_index][1] > 0):
                score_yes = score_yes + math.log10(conditionals[0][conditional_index][1])
                score_no = score_no + math.log10(conditionals[1][conditional_index][1])
        if (score_yes > score_no):
            prediction.append("yes")
            scores.append(score_yes)
        else:
            prediction.append("no")
            scores.append(score_no)
    
    return prediction, scores

def metrics(y_true, y_pred, y_pred_score, test_dataset):
    precision_no = precision_score(y_true, y_pred, pos_label="no")
    precision_yes = precision_score(y_true, y_pred, pos_label="yes")
    recall_no = recall_score(y_true, y_pred, pos_label="no")
    recall_yes = recall_score(y_true, y_pred, pos_label="yes")
    f1_no = f1_score(y_true, y_pred, pos_label="no")
    f1_yes = f1_score(y_true, y_pred, pos_label="yes")
    
    accuracy = accuracy_score(y_true, y_pred)
    
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    
    output_trace_file = open("../output/trace_NB-BOW-OV.txt", "w")
    for id, predict, score, result in zip(test_dataset[:,0], y_pred, y_pred_score, test_dataset[:,2]):
        output_trace_file.writelines(('{}  {}  {}  {}  {}\r').format(id, predict, score, result,  ("correct" if (predict == result) else "wrong")))
    output_trace_file.close()
    
    output_eval_file = open("../output/eval_NB-BOW-OV.txt", "w")
    output_eval_file.write("Accuracy_score: " + accuracy.astype(str))
    output_eval_file.write("\nprecision_score yes: " + precision_yes.astype(str) + "\tprecision_score no: " + precision_no.astype(str))
    output_eval_file.write("\nrecall_score yes: " + recall_yes.astype(str) + "\trecall_score no: " + recall_no.astype(str))
    output_eval_file.write("\nf1_score yes: " + f1_yes.astype(str) + "\tf1_score no: " + f1_no.astype(str))
    output_eval_file.close()

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
    prediction, scores = class_prediction(yes, no, conditionals, test_text, matches)
    
    metrics(test_q1_label, prediction, scores, test_dataset)
    print(scores)

# processing datasets
training = '../data/covid_training.tsv'
testing = '../data/covid_test_public.tsv'
data_processing(training, testing, 0.01)