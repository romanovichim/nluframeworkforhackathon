import pandas  as pd 
import yaml
import os 
import spacy
import string
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from pickle import dump


# Root directory of the project
ROOT_DIR = os.path.abspath(os.getcwd())

# Directory with nlu files
DATA_DIR = os.path.join(ROOT_DIR, "data")

#загрузим данные 
#ai_train = os.path.join(DATA_DIR, "atis_intents_train.csv")
ai_test = os.path.join(DATA_DIR, "atis_intents_test.csv")

ai_train  = os.path.join(DATA_DIR, "countryintents.csv")

import csv

def read_data(path):
    with open(path, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        labels = []
        sentences = []
        for row in readCSV:
            label = row[0]
            sentence = row[1]
            labels.append(label)
            sentences.append(sentence)
    return sentences, labels

# Loading Test Data

sentences_test,labels_test = read_data(ai_test)
print(sentences_test[:3],'\n')
print(labels_test[:3])

# Loading Training Data

sentences_train,labels_train = read_data(ai_train)
print(sentences_train[:3],'\n')
print(labels_train[:3])

# Load the spacy model: nlp
nlp = spacy.load('en_core_web_md')

#clean string
def preprocess_text(text):
    normalized_text = normalize(text)
    doc = nlp(normalized_text)
    removed_punct = remove_punct(doc)
    removed_stop_words = remove_stop_words(removed_punct)
    return lemmatize(removed_stop_words)

def normalize(text):
    # some issues in normalise package
    try:
        return ' '.join(normalise(text, verbose=False))
    except:
        return text

def remove_punct(doc):
    return [t for t in doc if t.text not in string.punctuation]

def remove_stop_words(doc):
    return [t for t in doc if not t.is_stop]

def lemmatize(doc):
    return ' '.join([t.lemma_ for t in doc])

def preproctextincolumn(sentence_list):
    for i in sentence_list:
        preprocess_text(i)
    return sentence_list

#preproctextincolumn(sentences_test)
#preproctextincolumn(sentences_train)

# Calculate the dimensionality of nlp
embedding_dim = nlp.vocab.vectors_length

print(embedding_dim)


def encode_sentences(sentences):
    # Calculate number of sentences
    n_sentences = len(sentences)

    print('Length :-',n_sentences)

    X = np.zeros((n_sentences, embedding_dim))
    #y = np.zeros((n_sentences, embedding_dim))

    # Iterate over the sentences
    for idx, sentence in enumerate(sentences):
        # Pass each sentence to the nlp object to create a document
        doc = nlp(sentence)
        # Save the document's .vector attribute to the corresponding row in     
        # X
        X[idx, :] = doc.vector
    return X

train_X = encode_sentences(sentences_train)
#test_X = encode_sentences(sentences_test)
# import labelencoder
from sklearn.preprocessing import LabelEncoder
# instantiate labelencoder object
le = LabelEncoder()

    
def label_encoding(labels,le):
    # Calculate the length of labels

    n_labels = len(labels)
    print('Number of labels :-',n_labels)

    y = le.fit_transform(labels)
    print(y[:100])
    print('Length of y :- ',y.shape)
    return y

train_y = label_encoding(labels_train,le)
#test_y = label_encoding(labels_test,le)


# Import SVC
from sklearn.svm import SVC
# X_train and y_train was given.
def svc_training(X,y):
    # Create a support vector classifier
    clf = SVC(C=1,probability=True)

    # Fit the classifier using the training data
    clf.fit(X, y)
    return clf

model = svc_training(train_X,train_y)


#Validation Step

def svc_validation(model,X,y):
    # Predict the labels of the test set
    y_pred = model.predict(X)

    # Count the number of correct predictions
    n_correct = 0
    for i in range(len(y)):
        if y_pred[i] == y[i]:
            n_correct += 1

    print("Predicted {0} correctly out of {1} training examples".format(n_correct, len(y)))


#svc_validation(model,train_X,train_y)
#svc_validation(model,test_X,test_y)

#from sklearn.metrics import classification_report
#y_true, y_pred = test_y, model.predict(test_X)
#print(classification_report(y_true, y_pred))


new_example=['hello']

def newstring(model,le,new_example):
    ex_to_fit=encode_sentences(new_example)
    ynew = model.predict(ex_to_fit)
    yproba = model.predict_proba(ex_to_fit)
    return ynew, le.inverse_transform(ynew),yproba[0][ynew[0]]



ns = newstring(model,le,new_example)
#ns[0]
print(ns,new_example)

new_example1=['show me the ground transportation']
ns1 = newstring(model,le,new_example1)
#ns[0]
print(ns1,new_example1)

# save the model
dump(model, open('model.pkl', 'wb'))
# save the le
dump(le, open('le.pkl', 'wb'))

