#   IF U USE THIS CODE IT WILL NOT RECOGNISE IT WILL ONLY RECOGNISE PRODUCT THAT ARE PRESENT IN THE PATTERN THATS NOT MY GOAL GOAL AFTER CATEGORY 
#SLECTION USER CAN PUT ANY PRODUCT NAME THAT HE WANTS.
import json 
import numpy as np
from tensorflow import keras
import colorama 
colorama.init()
from colorama import Fore, Style, Back
import pickle
import logging


with open("data.json") as file:
   data = json.load(file)
logging.info("imported json file for chatbot")

def chat():

    # load trained model
    model = keras.models.load_model('artifacts/chat_model.h5')
    logging.info("loaded chat model for chat")

    # load tokenizer object
    with open('artifacts/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    logging.info("tokkenizer loaded for tokkenze text")

    # load label encoder object
    with open('artifacts/label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
    logging.info("lablel loaded successfully")

    # parameters
    max_len = 20

   


    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()

        if inp.lower() == "quit":
            break

        # The user has selected a category, now handle product name
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                            truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])[0]

        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL, np.random.choice(i['response']))
                return ["I'm sorry, I didn't understand that. Could you please rephrase or ask another question?"]


#search chatbot who learns from its user'''


if __name__ == '__main__': 
    chat()
