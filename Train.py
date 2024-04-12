import json
import os
import logging
import numpy as np
import pickle
import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
file_path = 'D:\AI Bot\data.json'
with open(file_path) as file:
    data = json.load(file)
logging.info("imported json data to preprocess")

training_sentence = []
training_labels = list()
labels = []
responses = []

for intent in data['intents']:
    for inputs in intent['input']:
      training_sentence.append(inputs)
      training_labels.append(intent['tag'])
    responses.append(intent['response'])
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'])  
logging.info("Created seperated file for my training sentences and training labels and label and reponses")

num_classes = len(labels)

lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)
logging.info("Converted labels into numbers of  training labels")

##my unique words in sentences
vocab_size = 1000
##25 will be my  features
embedding_dim = 25
#this for padding max to max 21 words user can put 
max_len = 20

oov_token = "<OOV>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token = oov_token )
tokenizer.fit_on_texts(training_sentence)

word_index = tokenizer.word_index
sequence = tokenizer.texts_to_sequences(training_sentence)
padded_sequences = pad_sequences(sequence, truncating='post',maxlen=max_len)
#print(padded_sequences)
logging.info("Done Creating Embedding Layer")


logging.info("Stared Creating Model")
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
model.add(LSTM(128, input_shape=(max_len, embedding_dim)))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
logging.info("Compiled the model")

model.summary()
epochs = 70
logging.info("Started training the model")  
model.fit(padded_sequences,np.array(training_labels), epochs=epochs)
    
artifacts_folder = "artifacts"
os.makedirs(artifacts_folder, exist_ok=True)

# Save the trained model to the "artifacts" folder

model_filename = os.path.join(artifacts_folder, 'chat_model.h5')
model.save(model_filename)
logging.info('Model saved to ' + model_filename)

#Save the trained model to the "artifacts" folder

tokenizer_filename = os.path.join(artifacts_folder, 'tokenizer.pickle')
with open(tokenizer_filename,'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer saved to {tokenizer_filename}")
logging.info("Tokenizer saved to {tokenizer_filename}")   

#saving the fitted label encoder
# save the label encoder object to the "artifacts" folder

lbl_encoder_filename = os.path.join(artifacts_folder, 'label_encoder.pickle')
with open(lbl_encoder_filename, 'wb') as enc:
    pickle.dump(lbl_encoder,enc,protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Label Encoder saved to{lbl_encoder_filename}")
logging.info("label saved to{lbl_encoder_filename}")    