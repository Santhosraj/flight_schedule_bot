import random 
import json
import pickle 
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD
#nltk.download('punkt_tab')
lem = WordNetLemmatizer()
intents = json.loads(open("D:/internship/project_1/intents.json",encoding="utf-8").read())

words = []
classes = []
docs = []

ignore_letters = ['?',"!",".",","]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        docs.append((word_list,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lem.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words,open("words.pkl",'wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training = []
output_empty = [0] * len(classes)

for doc in docs:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lem.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)



x_train = np.array([item[0] for item in training])
y_train= np.array([item[1] for item in training])

print("Training data shape:", x_train.shape)
print("Training labels shape:", y_train.shape)

model = Sequential()
model.add(Dense(128,input_shape =(len(x_train[0]),),activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(64,activation = "relu" ))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]),activation= "softmax"))

optim = SGD(lr = 0.01,momentum = 0.9,nesterov = True)
model.compile(loss  = "categorical_crossentropy",optimizer = optim,metrics=['accuracy'])
model.fit(x_train,y_train,epochs =100,batch_size = 5,verbose = True)
model.save("project_1/botmodel.keras")