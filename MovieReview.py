from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
import nltk
from keras.preprocessing.text import Tokenizer
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from pandas import DataFrame
from matplotlib import pyplot

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
    # split into tokens by white space
    tokens = doc.split()

    # remove punctuation from each token
    #table = string.maketrans(string.punctuation,string.whitespace)
    #tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
	# load doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# update counts
	vocab.update(tokens)

# load all docs in a directory
def process_docs(directory, vocab):
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# add doc to vocab
		add_doc_to_vocab(path, vocab)

# define vocab
vocab = Counter()
# add all docs to vocab
process_docs('../txt_sentoken/pos', vocab)
process_docs('../txt_sentoken/neg', vocab)
# print the size of the vocab
#print(len(vocab))
# print the top words in the vocab
#print(vocab.most_common(50))

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

# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
	# load the doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens)


# load all docs in a directory
def process_docs(directory, vocab):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		line = doc_to_line(path, vocab)
		# add to list
		lines.append(line)
	return lines

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load all training reviews
positive_lines = process_docs('../txt_sentoken/pos', vocab)
negative_lines = process_docs('../txt_sentoken/neg', vocab)
# summarize what we have
#print(len(positive_lines), len(negative_lines))

# fit the tokenizer on the documents
train_docs = positive_lines + negative_lines

# load all docs in a directory
def process_docs_test(directory, vocab, is_trian):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		line = doc_to_line(path, vocab)
		# add to list
		lines.append(line)
	return lines

# load all test reviews
positive_lines = process_docs_test('../txt_sentoken/pos', vocab, False)
negative_lines = process_docs_test('../txt_sentoken/neg', vocab, False)
test_docs = positive_lines+negative_lines



ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])

ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])


# create the tokenizer
tokenizer = Tokenizer()
# prepare bag of words encoding of docs
def prepare_data(train_docs, test_docs, mode):

	# fit the tokenizer on the documents
	tokenizer.fit_on_texts(train_docs)
	# encode training data set
	Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
	# encode training data set
	Xtest = tokenizer.texts_to_matrix(test_docs, mode=mode)
	return Xtrain, Xtest



# evaluate a neural network model
def evaluate_mode(Xtrain, ytrain, Xtest, ytest):
	scores = list()
	n_repeats = 1
	n_words = Xtest.shape[1]
	for i in range(n_repeats):
		# define network
		model = Sequential()
		model.add(Dense(50, input_shape=(n_words,), activation='relu'))
		model.add(Dense(1, activation='sigmoid'))
		# compile network
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		# fit network
		model.fit(Xtrain, ytrain, epochs=5, verbose=0)
		# evaluate
		loss, acc = model.evaluate(Xtest, ytest, verbose=2)
		scores.append(acc)
		print('%d accuracy: %s' % ((i+1), acc))
	return model

modes = ['freq']
#results = DataFrame()
for mode in modes:
	# prepare data for mode
	Xtrain, Xtest = prepare_data(train_docs, test_docs, mode)
	# evaluate model on data for mode
	results = evaluate_mode(Xtrain, ytrain, Xtest, ytest)
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
	return round(yhat[0,0])

# test positive text
text = 'Best movie ever!'
print(predict_sentiment(text, vocab,tokenizer, results))
# test negative text
text = 'This is a really bad movie useless.'
print(predict_sentiment(text, vocab,tokenizer, results))