import json
from keras.models import model_from_json # to load model
from keras.preprocessing.text import Tokenizer # tokenization
from keras.layers import Input # input layer
#from BaselineModels import get_f1 # to calculate f1 score
#import aoc_id.data_helpers as dh # for importing data
import numpy as np
import keras.backend as K # to calculated f1_score

def get_f1(y_true, y_pred): #taken from old keras source code
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  recall = true_positives / (possible_positives + K.epsilon())
  f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
  return f1_val


# alternative to data_helpers.LoadPRED without normalization
def loadPretrained(data):
	df = open(data, 'r')
	sentences = []
	for line in df:
		sentences.append(line)
	return sentences

# load model
with open('bigru_binary_10_epochs.json', 'r') as json_file:
    architecture = json.load(json_file)
    model = model_from_json(json.dumps(architecture))

# load weights and compiling model
model.load_weights('bigru_binary_10_epochs.h')
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=[get_f1])

# import data to filter
sentences = loadPretrained('tweets_mixed.txt')

# prepare data for input

def tokenizeData(data): 
    #init tokenizer
    tokenizer= Tokenizer(filters='\t\n',split=" ",char_level=False)
    #use tokenizer to split vocab and index them
    tokenizer.fit_on_texts(data)
    # txt to seq
    data = tokenizer.texts_to_sequences(data)
    
    return data

tok_sent = tokenizeData(sentences)

arr = np.array(tok_sent)
np.array(arr[-1]).shape

for s in range(len(tok_sent)):
  for i in range(100-len(tok_sent[s])):
    tok_sent[s].append(0)

# check for problems in input size

for s in tok_sent:
  safe = False
  if len(s) != 100:
    print("Problem detected!")
  else:
    safe = True
if safe:
  print("No issues with input size.")

# classify sentences 

prediction = model.predict(np.reshape(tok_sent[i:i+1], (1,100)))

lev_file = open('Filtered_LEV.txt', 'w+')
other_file = open('Filtered_NOT_LEV.txt', 'w+')

for i in range(len(tok_sent)):
  prediction = model.predict(np.reshape(tok_sent[i:i+1], (1,100)))
  if float(prediction[0][0]) >= float(prediction[0][1]):
    lev_file.write(sentences[i])
  else:
    other_file.write(sentences[i])
  
print("Done")
