import pandas  as pd 
import os 
import spacy
import csv
import string
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from pickle import load

# Root directory of the project
ROOT_DIR = os.path.abspath(os.getcwd())

# Directory with nlu files
DATA_DIR = os.path.join(ROOT_DIR, "data")

#CApital Country dataset
capitalcountry = os.path.join(DATA_DIR, "countrycapital.csv")

#Answer dataset
answer = os.path.join(DATA_DIR, "countryanswers.csv")

#Model and le
MODEL_DIR = os.path.join(ROOT_DIR, "intentmodel")

#Model
model_path = os.path.join(MODEL_DIR, "model.pkl")
#le
le_path = os.path.join(MODEL_DIR, "le.pkl")

#NER
nlp = spacy.load("en_core_web_md")

#doc = nlp("Ada Lovelace was born in London Canada London Ottawa")

# document level
#for ent in doc.ents:
#    print(ent.text, ent.label_)


def read_data(path):
    with open(path, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        capcountry = []
        for row in readCSV:
            label = [row[0],row[1]]
            capcountry.append(label)
            
    return capcountry


def read_dict(path):
    with open(path, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        capcountry = []
        mydict = {rows[0]:rows[1] for rows in readCSV}
            
    return mydict

#Тестовый кейсы
# Два Ner - Страна и город
test1 ="Ottawa is the capital of Canada"
# Один Ner
test2 ="Peru"
# Intent
test3 ="hello"
# Intent Nomatch
test4 ="Make me coffee please"


testlist = ["Ottawa is the capital of Canada","Peru","hello","Make me coffee please"]

def nerintstr(string,nlp):
    doc = nlp(string)
    ner_list=[]
    for ent in doc.ents:
        #текст ner текст лейбла
        temp = [ent.text, ent.label_]
        ner_list.append(temp)

    return ner_list

def lowstr(string):
    res = string.lower()
    return res


def findmatchreturnrow(twolist,match_str):
    t_res = None
    for x, tlist in enumerate(twolist):
        for strr in tlist:
            if match_str in strr:
                t_res = x   

    return t_res

def encode_sentences(sentences):
    # Calculate number of sentences
    n_sentences = len(sentences)

    #print('Length :-',n_sentences)
    embedding_dim = nlp.vocab.vectors_length
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


def test_trans(s):
    table = string.maketrans("","")
    return s.translate(table, string.punctuation)


def newstring(model,le,new_example):
    ex_to_fit=encode_sentences(new_example)
    ynew = model.predict(ex_to_fit)
    yproba = model.predict_proba(ex_to_fit)
    return ynew, le.inverse_transform(ynew),yproba[0][ynew[0]]


def intentrecognition(model,le,new_example,confidence=0.75):
    tres = 'nomatch'
    ns = newstring(model,le,new_example)
    if(ns[2] > confidence):
        tres = ns[1][0]
    return tres    
    
def childrendialog(string,nlp,answer):
    # load the model
    model = load(open(model_path, 'rb'))
    # load the scaler
    le = load(open(le_path, 'rb'))
    answer_dict = read_dict(answer)
    result = answer_dict['nomatch']
    templist = nerintstr(string,nlp)
    templen = len(templist)
    if(templen > 1):
        first_temp = findmatchreturnrow(read_data(capitalcountry),templist[0][0])
        second_temp = findmatchreturnrow(read_data(capitalcountry),templist[1][0])
        if(first_temp is not None and second_temp is not None):
            if(first_temp == second_temp):
                # Правильно
                result = "YEAH that's true let's try another pair!"
            else:
                # Неправильно
                result = "NO nonono that's false let's try another pair!"
        else:
            result = answer_dict['nomatch']  
    elif(templen == 1):
        result = answer_dict['nomatch']
    elif(templen == 0):
        #если не распознали ничего то смотрим по интентам
        result = answer_dict[intentrecognition(model,le,[string])]
    else:
        result = answer_dict['nomatch']
    
    return result

# Цикл для обработки

for text in testlist:
    print("На вход: ", text)
    print("На выход: ", childrendialog(text,nlp,answer))


#print('-------')
#answer_dict = read_dict(answer)
#print(answer_dict['nomatch'])








#print("------")


#new_example=['hello']
#ns = newstring(model,le,new_example)
#ns[0]
#print(ns[1][0],ns[2],new_example)
#t = intentrecognitionD(model,le,new_example)
#print(t)

#new_example1=['show me the ground transportation']
#ns1 = newstring(model,le,new_example1)
#ns[0]
#print(ns1,new_example1)
#t = intentrecognitionD(model,le,new_example1)
#print(t)







