import collections

# a 2d list of training sets, where each index should contains array of 3 elements
train_set_2D_list = []

# read file with utf-8
train_file = open("data/covid_training.tsv", "r", encoding="utf-8")
train_list = train_file.readlines()
train_file.close()
# remove headear row
del train_list[0]
# put every training set with tweet_id, text, and q1_label into 2d array
for row in train_list:
    train_string_list = str(row).split("\t")
    train_set_2D_list.append([train_string_list[0], train_string_list[1].lower(), train_string_list[2]])
# class collections.Counter to count word in all tweet texts
word_collections_counter = collections.Counter()
for row in train_set_2D_list:
    word_collections_counter.update(row[1].split(" "))
# Filter out words that appear only once in the training set
for key, value in list(word_collections_counter.items()):
    if value == 1:
        del word_collections_counter[key]
# a filtered two-dimensional list (list of tuple objects with word and word frequencies)
filtered_vocabulary_list = list(word_collections_counter.items())
