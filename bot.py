import random
import json
import pickle
import numpy as np 
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import requests
from datetime import datetime, timedelta
from pytz import timezone
import openapi_client
import aiohttp
import asyncio


lemmatizer = WordNetLemmatizer()
intents = json.load(open("intents.json", encoding="utf-8"))
words = pickle.load(open(r"words.pkl", 'rb'))
classes = pickle.load(open(r"classes.pkl", 'rb'))
model = load_model(r"botmodel.keras")
        
def clean_sentence(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

def bag_of_words(sentence):
        sentence_words = clean_sentence(sentence)
        bag = [0] * len(words)
        for w in sentence_words:
            for i, word in enumerate(words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

def extract_cities(message):
        message = message.lower()
        from_city = to_city = None

        try:
            if "from" in message and "to" in message:
              
                parts = message.split("from")[1].split("to")
                if len(parts) >= 2:
                    from_city = parts[0].strip().upper()
                    to_city = parts[1].strip().upper()
                    from_city = from_city.split()[0] if from_city else None
                    to_city = to_city.split()[0] if to_city else None

        except Exception as e:
            print(f"Error in city extraction: {str(e)}")
            return None, None

        return from_city, to_city

def get_flight_data(from_city, to_city):
        AUTH_ENDPOINT = "https://test.api.amadeus.com/v1/security/oauth2/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {"grant_type": "client_credentials",
                "client_id": '',
                "client_secret": ''}
        response = requests.post(AUTH_ENDPOINT,
                                headers=headers,
                                data=data)
        access_token = response.json()['access_token']
        headers = {'Authorization': 'Bearer' + ' ' + access_token}
        
        flight_url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
        current_date = datetime.now().strftime("%Y-%m-%d")
        params = {
            "originLocationCode": from_city,  
            "destinationLocationCode": to_city,    
            "departureDate": current_date , 
            "max": 5,
            "adults":1
        }
        
        try:
        
            response = requests.get(flight_url, params=params,headers=headers)
            print(f"API Response Status: {response.status_code}")  
            
            if response.status_code != 200:
                print(f"API Error Response: {response.text}")  
                return None
                
            data = response.json()
            print(f"API Response Data: {data}") 
            
            if 'data' in data and len(data['data']) > 0:
                flights = []
                for flight in data['data']:
                    flight_info = {
                        'airline': flight.get('airline', {}).get('name'),
                        'flight_number': f"{flight.get('flight', {}).get('iata')}",
                        'departure': flight.get('departure', {}).get('scheduled'),
                        'arrival': flight.get('arrival', {}).get('scheduled'),
                        'status': flight.get('flight_status'),
                       
                    }
                    flights.append(flight_info)
                return flights
            else:
                print("No flight data found in response") 
                return None
                    
        except Exception as e:
            print(f"Exception in get_flight_data: {str(e)}") 
            return {"error": str(e)}

def predict(sentence):
        bow = bag_of_words(sentence)
        res = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25

        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)

        return_list = []
        for r in results:
            return_list.append({
                "intent": classes[r[0]],
                "probability": str(r[1])
            })

        return return_list

def format_flight_response(flights):
        if not flights:
            return "No flights found for the specified route."
        
        if isinstance(flights, dict) and 'error' in flights:
            return f"Error fetching flight information: {flights['error']}"

        try:
            response = "Here are the available flights:\n\n"
            for flight in flights:
                departure = flight.get('departure')
                arrival = flight.get('arrival')
                
                if departure and arrival:
                    departure_dt = datetime.fromisoformat(departure.replace('Z', '+00:00'))
                    arrival_dt = datetime.fromisoformat(arrival.replace('Z', '+00:00'))
                    
                    response += f"Airline: {flight.get('airline', 'N/A')}\n"
                    response += f"Flight Number: {flight.get('flight_number', 'N/A')}\n"
                    response += f"Departure: {departure_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    response += f"Arrival: {arrival_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    response += f"Status: {flight.get('status', 'N/A')}\n"
                    response += "-" * 40 + "\n"
            
            return response

        except Exception as e:
            print(f"Error formatting flight response: {str(e)}")
            return "Error formatting flight information."

def get_response(message):
        intent_list = predict(message)      
        tag = intent_list[0]['intent']

        if any(word in message.lower() for word in ['flight', 'flights', 'from', 'to']):
            from_city, to_city = extract_cities(message)
            
            if from_city and to_city:
                print(f"Searching flights from {from_city} to {to_city}") 
                flights = get_flight_data(from_city, to_city)
                if flights and 'error' not in flights:
                    return format_flight_response(flights)
                else:
                      return "Sorry, I couldn't fetch flight for the given cities.Kindly check the IATA code"
            else:
                return "Please specify both departure and arrival cities clearly. For example: 'Show flights from MAA to CJB'"
    
        for intent in intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
        
        return "OOPS! IDK😅"

def main():
    
    print("Bot: Hello, I am AirMate✈️! How can I help you today? (Type 'quit' to exit)")
    
    while True:
        message = input("You: ")
        if message.lower() == 'quit':
            print("Bot: Goodbye! Have a great day!")
            break
            
        response =get_response(message)
        print("Bot:", response)

if __name__ == "__main__":
    main()
