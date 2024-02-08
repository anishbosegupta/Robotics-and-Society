#Import files
import os,random,json,torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import spacy
import datetime
import requests
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("intents.json") as json_data:
    intents = json.load(json_data)

data_dir = os.path.join(os.path.dirname(__file__))
FILE = os.path.join(data_dir, 'chatdata.pth')
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

#Load the NLP model
nlp = spacy.load("en_core_web_sm")

# Define a function to get the weather using the weather API
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

bot_name = "Tony"
def get_response(msg):
    doc = nlp(msg.lower())
    #print("doc",doc)
    sentence = tokenize(msg)
    #print("sentence ",sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
   
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            #print("intent",intent)
            if tag == intent["tag"]:
                #print("tag",tag)
                if tag == "time":
                    current_time = str(datetime.datetime.now().strftime("%H:%M"))
                    return random.choice(intent['responses']) + " " + current_time
                elif tag == "date":
                    return get_date()
                elif tag == "weather":
                    for token in doc:
                        #print("token",token)
                    #if token.text in ["in", "of"]:
                        city = [entity.text for entity in doc.ents if entity.label_ == "GPE"]
                        #print("city", city)
                        if city:
                            return get_weather(city[0])
                        else: return "Please specify a city to get the weather for."                 
                else:
                    return random.choice(intent['responses'])
    return "I do not understand..."
if __name__ == "__main__":
    print("Hi! This is Tony, your assistant. How can I help you today? (type 'quit' to exit)")
    while True:
        sentence = input("Anish: ")
        if sentence == "quit":
            print("Bye now!")
            break
        resp = get_response(sentence)
        print(resp)

