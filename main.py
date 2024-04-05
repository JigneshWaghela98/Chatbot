

import json 
import numpy as np
from tensorflow import keras
import colorama 
colorama.init()
from colorama import Fore, Style, Back
import pickle
import logging
from flask import Flask, request, jsonify
import speech_recognition as sr  
import pyttsx3
import pyaudio

with open("data.json") as file: 
   data = json.load(file)
logging.info("imported json file for chatbot")

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

def chat(question):
        inp = question
        max_len=20
            
        # The user has selected a category, now handle product name
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                            truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])[0]

        for i in data['intents']:
            if i['tag'] == tag:
                response= np.random.choice(i['response'])
                print(response)
                break
        else:
            # This block will execute if no matching tag is found
            print("I'm sorry, I didn't understand that. Could you please rephrase or ask another question?")
            response = "I'm sorry, I didn't understand that. Could you please rephrase or ask another question re?"

        return response

engine = pyttsx3.init()
# Set the speaking rate (adjust as desired)
engine.setProperty('rate', 180)

# Set the volume level (between 0 and 1)
engine.setProperty('volume', 1.0)
voices = engine.getProperty('voices')       #getting details of current voice
#engine.setProperty('voice', voices[0].id) 
engine.setProperty('voice', voices[1].id)   #changing index, changes voices. o for male
    
def recognize_speech_and_respond():
    # initialized the recognizer
    recognizer = sr.Recognizer()

    # capturing voice command
    with sr.Microphone() as source:
        print("Listening for a question about India and Its Culture...")
        audio = recognizer.listen(source)

    # Recognize speech using Google Web Speech API
    try:
        question = recognizer.recognize_google(audio)
        print(f"Question: {question}")
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return

    
    


    # Get the response from the AI model
    response = chat(question)
    print(f"Response: {response}")

    engine.say(response)
    engine.runAndWait()
    engine.stop()

# Example usage
if __name__ == "__main__":
    recognize_speech_and_respond()