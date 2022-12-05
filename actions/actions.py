# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
#        dispatcher.utter_message(text=answer)


from typing import Any, Dict, List, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.events import UserUtteranceReverted
from rasa_sdk.executor import CollectingDispatcher

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import matplotlib.pyplot as plt
import numpy as np
from PyPDF2 import PdfReader


tokenizer = AutoTokenizer.from_pretrained("./actions/bert/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",local_files_only=True)
model = AutoModelForQuestionAnswering.from_pretrained("./actions/bert/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",local_files_only=True)
reader = PdfReader("./actions/bert/contenido.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"


#print(text)

class ActionHelloWorld(Action):
    def name(self) -> Text:
        return "action_hello_world"
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="Hello World!")
        return []


class ActionDefaultFallback(Action):
    def name(self) -> Text:
        return "action_default_fallback"

    def run(self,dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],) -> List[Dict[Text, Any]]:
        message1 = tracker.latest_message.get('text')  
        print(message1)
        questions = [
            #"¿Cuál es mi nombre?"
            message1
        ]        
        for question in questions:
            inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
            input_ids = inputs["input_ids"].tolist()[0]
        
            text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            answer_start_scores, answer_end_scores = model(**inputs)[0], model(**inputs)[1]
        
            answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
        
            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
            print(f"Question: {question}")
            print(f"Answer: {answer}\n")
            dispatcher.utter_message(text=answer)

        

        return []


class BertResponse(Action):
    def name(self) -> Text:
        return "call_bert"

    def run(self,dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],) -> List[Dict[Text, Any]]:
        
        text1 = r"""
        Mi nombre es Jorge Auquilla y trabajo como científico de datos en Persado. 
        Mi blog personal se llama Predictive Hacks, que ofrece tutoriales principalmente en R y Python.
        """

        questions = [
            "¿Cuál es mi nombre?"
        ]
        
        for question in questions:
            inputs = tokenizer.encode_plus(question, text1, add_special_tokens=True, return_tensors="pt")
            input_ids = inputs["input_ids"].tolist()[0]
        
            text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            answer_start_scores, answer_end_scores = model(**inputs)[0], model(**inputs)[1]
        
            answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
        
            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        
        print(f"Question: {question}")
        print(f"Answer: {answer}\n")
        
        dispatcher.utter_message(text=answer)
        
        return []


