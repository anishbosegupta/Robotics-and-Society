import spacy
import random
import datetime
import requests
import warnings
warnings.filterwarnings('ignore')

#Load the NLP model
nlp = spacy.load("en_core_web_sm")

#Define the intents and corresponding responses
intents = {
    "greet": ["Hello!", "Hi!", "Hey!", "Howdy!"],
    "goodbye": ["Goodbye!", "See you later!", " Bye!"],
    "thank you": ["You're welcome!", "No problem!", "My pleasure!"],
    "get cup": ["Sure, here is your cup."],
    "time": ["The current time is:"],
    "weather": ["I am not capable of checking the weather, sorry."],
    "date": ["Today is {0}.".format(datetime.datetime.now().strftime("%B %d, %Y"))],
    "joke": ["What do you call a lazy kangaroo? A pouch potato!"]
}

# Define a function to get the weather
def get_weather(city):
    try:
        # Use an API to get the weather information for the specified city
        response = requests.get(f"http://api.openweathermap.org/data/2.5/weather?units=imperial&q={city}&appid=d051ee0e8bff5385a4b16d0b67cae65f")
        data = response.json()
        if data["cod"] == "404":
            return "City not found, please try again."
        else:
            # Extract and format the relevant weather information
            weather = data["weather"][0]["description"]
            temperature = data["main"]["temp"]
            return f"The weather in {city} is currently {weather} with a temperature of {temperature:.2f}°F."
    except requests.exceptions.RequestException as e:
        return "An error occurred while trying to get the weather information. Please try again later."

# Define a function to get the current date
def get_date():
    #return str(datetime.datetime.now().date())
    return "Today is {0}.".format(datetime.datetime.now().strftime("%B %d, %Y"))

# Define the chatbot function
def chatbot(message):
    # Use the NLP model to process the message
    #print(f"User: {message}")
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    doc = nlp(message.lower())
    #print("doc",doc)
    result = []
    for token in doc:
        if token.text in stop_words:
            continue
        if token.is_punct:
            continue
        if token.lemma_ == '-PRON-':
            continue
        result.append(token.lemma_)
    message_clean = " ".join(result)
    #print(f"This is the cleaned message: {message_clean}")
    doc_clean = nlp(message_clean)
    # Determine the user's intent
    for intent, responses in intents.items():
        doc_intent = nlp(intent)
        similarity = doc_clean.similarity(doc_intent)
        #print(similarity)
        if intent in [token.text for token in doc_clean] or similarity > 0.75:
            if intent == "time":
                current_time = str(datetime.datetime.now().strftime("%H:%M"))
                return responses[0] + " " + current_time
            elif intent == "date":
                return get_date()
            elif intent == "weather":
                for token in doc:
                    #if token.text in ["in", "of"]:
                    city = [entity.text for entity in doc.ents if entity.label_ == "GPE"]
                    #print("city", city)
                    if city:
                        return get_weather(city[0])
                return "Please specify a city to get the weather for."
            return random.choice(responses)
    
    # If the intent can't be determined, return a default response
    return "I'm sorry, I don't understand what you're saying."


if __name__ == "__main__":
    print("Hi! This is Tony, your assistant. How can I help you today? (type 'quit' to exit)")
    while True:
        sentence = input("Anish: ")
        if sentence == "quit":
            print("Bye now!")
            break
        resp = chatbot(sentence)
        print(resp)