import numpy as np
import pandas as pd


def bag_of_words(features, targets):
    bag = {
        'ham': dict(),
        'ham_num_words': 0,
        'spam': dict(),
        'spam_num_words': 0
    }

    for index in range(len(targets)):
        email = features[index]
        pocket = targets[index]

        for word in email.split(" "):
            word = word.lower()
            if word in bag[pocket].keys():
                bag[pocket][word] = bag[pocket][word] + 1
            else:
                bag[pocket][word] = 2  # avoid zeros

            if pocket == 'ham':
                bag['ham_num_words'] = bag['ham_num_words'] + 1
                if word not in bag['spam']:
                    bag['spam'][word] = 1  # avoid zeros
            elif pocket == 'spam':
                bag['spam_num_words'] = bag['spam_num_words'] + 1
                if word not in bag['ham']:
                    bag['ham'][word] = 1  # avoid zeros

    bag['ham_num_words'] = bag['ham_num_words'] + len(bag['ham'])
    bag['spam_num_words'] = bag['spam_num_words'] + len(bag['spam'])

    return bag


def bag_with_words_and_probs(bag, targets):
    number_of_emails = len(targets)
    number_of_ham = targets.value_counts().get('ham')
    number_of_spam = number_of_emails - number_of_ham

    pocket = 'ham'
    words = bag[pocket]
    for word in words:
        bag[pocket][word] = (words[word], words[word] / bag['ham_num_words'])
    bag[pocket] = (words, number_of_ham / number_of_emails)

    pocket = 'spam'
    words = bag[pocket]
    for word in words:
        bag[pocket][word] = (words[word], words[word] / bag['spam_num_words'])
    bag[pocket] = (words, number_of_spam / number_of_emails)

    return bag


def predict(model, email):
    email = email.split(" ")

    ham_words, ham_prob = model['ham']
    for word in email:
        word = word.lower()
        word_prob = ham_words[word][1] if word in ham_words else (1 /
                                                                  (len(ham_words) + 1))  # avoid zeros
        ham_prob = ham_prob * word_prob
    print('ham_score: ' + str(ham_prob))

    spam_words, spam_prob = model['spam']
    for word in email:
        word = word.lower()
        word_prob = spam_words[word][1] if word in spam_words else (1 /
                                                                    (len(spam_words) + 1))  # avoid zeros
        spam_prob = spam_prob * word_prob
    print('spam_score: ' + str(spam_prob))

    if ham_prob > spam_prob:
        return 'ham'
    elif spam_prob > ham_prob:
        return 'spam'
    else:
        return 'undecided'


if __name__ == '__main__':
    data_url = "spam.csv"
    raw_dataframe = pd.read_csv(data_url, sep=",")

    X_train = raw_dataframe.drop(['v1'], axis=1).astype(str).agg(" ".join, axis=1)
    y_train = raw_dataframe['v1']

    bag = bag_of_words(X_train, y_train)
    model = bag_with_words_and_probs(bag, y_train)

    x_test = "You have 1 new message. Call 0207-083-6089"
    prediction = predict(model, x_test)
    print()
    print(prediction)
