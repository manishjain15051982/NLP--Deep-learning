from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
import nltk
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization,Input, Activation
from keras.models import Model
from pandas import DataFrame
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# turn a doc into clean tokens
def clean_doc(doc):
    doc = preprocess(doc)
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    #table = str.maketrans(str.punctuation,' ')
    #tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

def preprocess(text):
    text = text.replace("' ", " ' ")
    signs = set(',.:;"?!')
    prods = set(text) & signs
    if not prods:
        return text

    for sign in prods:
        text = text.replace(sign, ' {} '.format(sign) )
    return text

# load doc and add to vocab
def add_doc_to_vocab(X_train, vocab):
	# clean doc
    for doc in X_Train["text"]:
	    tokens = clean_doc(doc)
	    # update counts
	    vocab.update(tokens)



# define vocab
vocab = Counter()

X_Train= pd.read_csv("train.csv")

# add all docs to vocab
add_doc_to_vocab(X_Train,vocab)

# print the size of the vocab
#print(len(vocab))
# print the top words in the vocab
#print(vocab.most_common(100))


# keep tokens with a min occurrence
min_occurane = 2
tokens = [k for k,c in vocab.items() if c >= min_occurane]
#print(len(tokens))


# save list to file
def save_list(lines, filename):
    # convert lines to a single blob of text
    data = '\n'.join(lines)
    # open file
    file = open(filename, 'w')
    # write text
    file.write(data)
    # close file
    file.close()

# save tokens to a vocabulary file
save_list(tokens, 'vocab.txt')

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

train_docs = X_Train["text"]
y_train = np.array(pd.get_dummies(X_Train["author"]))




#print (len(train_docs))

# create the tokenizer
tokenizer = Tokenizer()
# prepare bag of words encoding of docs
def prepare_data(train_docs, mode):

	# fit the tokenizer on the documents
	tokenizer.fit_on_texts(train_docs)
	# encode training data set
	Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
	return Xtrain




# evaluate a neural network model
def evaluate_mode(Xtrain, ytrain):
    scores = list()
    n_repeats = 1
    n_words = Xtrain.shape[1]
    for i in range(n_repeats):

        #model = Sequential()
        #model.add(Dense(500, input_shape=(n_words,),activation='relu'))
        #model.add(Dense(50, input_shape=(n_words,),activation='relu'))
        #model.add(Dense(3, activation='softmax'))

        X_input = Input(shape=(n_words,))
        # Dense -> BN -> RELU->dropout Block applied to X
        X = Dropout(0.5)(X_input)
        X = Dense(900, kernel_initializer='he_normal', name='D0')(X)
        X = BatchNormalization(axis=1, name='bn0')(X)
        X = Activation('relu')(X)
        X = Dropout(0.5)(X)

        X = Dense(600, kernel_initializer='he_normal', name='D1')(X)
        X = BatchNormalization(axis=1, name='bn1')(X)
        X = Activation('relu')(X)
        X = Dropout(0.5)(X)

        X = Dense(300, kernel_initializer='he_normal', name='D2')(X)
        X = BatchNormalization(axis=1, name='bn2')(X)
        X = Activation('relu')(X)
        X = Dropout(0.5)(X)

        X = Dense(3, kernel_initializer='he_normal', activation='softmax')(X)

        Spookymodel = Model(inputs=X_input, outputs=X, name='SpookyAuthor')

        # compile network
        Spookymodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fitnetwork
        Spookymodel.fit(x= Xtrain, y=ytrain, epochs=50,verbose=2,batch_size=256, validation_split=0.2)
        # evaluate

    return Spookymodel

modes = ['freq']
results = DataFrame()
for mode in modes:
    # prepare data for mode
    X_train = prepare_data(train_docs, mode)
    #print ("X_train", X_train_final)
    #print ("y_train", y_train_final)
    #X_train_final, X_test, y_train_final, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
    # evaluate model on data for mode
    model = evaluate_mode(X_train,y_train)

# summarize results
#print(results.describe())
# plot results
#results.boxplot()
#pyplot.show()

# classify a review as negative (0) or positive (1)
def predict_sentiment(review, vocab,tokenizer, model):
    # clean
    tokens = clean_doc(review)
    # filter by vocab
    tokens = [w for w in tokens if w in vocab]
    # convert to line
    line = ' '.join(tokens)
    # encode
    encoded = tokenizer.texts_to_matrix([line], mode='freq')
    # prediction
    yhat = model.predict(encoded, verbose=0)
    return yhat



X_sub = pd.read_csv("test.csv")

y_pred = np.zeros((X_sub.shape[0],3))

i = 0
for doc in X_sub["text"]:
    y_pred[i]=predict_sentiment(doc, vocab,tokenizer,model)
    i +=1



submission = pd.DataFrame(y_pred,dtype=float)
submission.insert(0,'id',X_sub["id"])
submission=submission.rename(index=int, columns={0: "EAP", 1: "HPL", 2: "MWS"})
#print(submission.head())


submission.to_csv("sub.csv",sep=',', encoding='utf-8')
