'''import pyttsx3
import speech_recognition as sr


engine = pyttsx3.init()
# Set the speaking rate (adjust as desired)
engine.setProperty('rate', 180)

# Set the volume level (between 0 and 1)
engine.setProperty('volume', 1.0)

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
    response = get_openai_response(question)
    print(f"Response: {response}")

    engine.say(response)
    engine.runAndWait()
    engine.stop()


# Example usage
if __name__ == "__main__":7
    recognize_speech_and_respond()'''
    

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
                #return ["I'm sorry, I didn't understand that. Could you please rephrase or ask another question?"]

        return response

engine = pyttsx3.init()
# Set the speaking rate (adjust as desired)
engine.setProperty('rate', 180)

# Set the volume level (between 0 and 1)
engine.setProperty('volume', 1.0)
    
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

   
app=Flask(__name__)#difining flask for api calling

@app.route('/textapi', methods=['POST'])#
def text_api():
    if request.method == 'POST':
        data = request.get_json()

        if 'text' in data:
            text = data['text']
            result = chat(text)
            return jsonify(result)
        else:
            return jsonify({'error': 'Text not provided in the request'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4567, debug=True)




