import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from datetime import datetime, date
from datetime import timedelta




with open("intents.json") as file:
    data = json.load(file)



# Chatbot initialization
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        # detecting frequency of "key words" from user inputs and store it into a ba (eg. [0,0,1,0,1,0])
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# initialize the ANN
tensorflow.reset_default_graph()

# input layer
net = tflearn.input_data(shape=[None, len(training[0])])
# 2 hidden layers
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
# output layer
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

try:
    model = tflearn.DNN(net)
    model.load("model.tflearn")
except:
    model = tflearn.DNN(net)
    # we train each set of input-output pairs 1000 times
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def breakSuggestionAlg(info):
    print("I suggest you to take a break with some good ol'", info["study_break"])


def chat(info):
    print("Hello", info["name"], "! SAFOU at your service")
    while True:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        # generate a response if the confidence is over 75%
        if results[results_index] > 0.75:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(current_time, random.choice(responses))
            if tag == "start_studying":
                print("also.. have fun :D")
                start_studying_time = now.time()
            elif tag == "break_suggestions":
                breakSuggestionAlg(info);
                end_studying_time = now.time()
                try:
                    study_time = datetime.combine(date.today(), end_studying_time) \
                             - datetime.combine(date.today(), start_studying_time)
                    print("You studied for a total of", study_time)
                except:
                    print("but.. you haven't started studying")
            elif tag == "major":
                print("edit here")
        else:
            print("Sorry, I don't understand.")




def introUI():
    try:
        with open("information.json") as file:
            info = json.load(file)
    except:
        name = input("What is your name? ")
        sex = input("What is your gender? (enter male or female) ")
        major = input("What is your specialization (major)? ")
        courses = input("What courses are you currently taking this term? ")
        # further mention  courses that the user is most/least confident in
        weekday = input("How many hours do you study on a weekday? ")
        weekend = input("How many hours do you study on a weekend? ")
        study_break = input("What do you like to do during study break? (less than 1 hour) ")
        free_time = input("What do you like to do during free time? (more than 1 hour) ")
        day_off = input("What do you like to do on a day off? ")


        info = {}
        info["name"] = name
        if sex == "male": # 0 as male, 1 as female
            info["sex"] = 0
        elif sex == "female":
            info["sex"] = 1
        info["major"] = major # range from 0 to 1, from memorization-based (0) to application-based (1)
        info["courses"] = courses
        info["weekday"] = int(weekday) # number of hours indicate studiousness
        info["weekend"] = int(weekend) # number of hours indicate studiousness
        info["study_break"] = study_break
        info["free_time"] = free_time # range from 0 to 1 to indicate "amount of bodily movement", from wathcing YouTube/Netflix (0) to playing sports (1)
        info["day_off"] = day_off # range from 0 to 1 to indicate "amount of bodily movement", from wathcing YouTube/Netflix (0) to playing sports (1)
        # note that all these inputs are Strings

        with open('information.json', 'w') as f:
            json.dump(info, f)

    chat(info)




def getMotivationalLevel(info):
    students = pd.read_csv("students.csv", sep=",")
    data = students[["sex","wkdy","wknd","molvl"]]
    toPredict = "molvl"
    x = numpy.array(data.drop([toPredict], 1))
    y = numpy.array(data[toPredict])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.5) # should modify test_size once the data set is large enough
    model = KNeighborsClassifier(n_neighbors=9)

    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(acc)

    predicted = model.predict([[info["sex"], info["weekday"], info["weekend"]]])
    print(predicted)








introUI()


