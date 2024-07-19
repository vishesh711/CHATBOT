# Chatbot Using Deep Learning

This project is a simple chatbot implemented using deep learning techniques. It leverages a Sequential Neural Network model to predict responses based on user input. The chatbot can understand and respond to various intents defined in a JSON file.

## Features

- Tokenizes and lemmatizes user input
- Uses a bag-of-words model for feature extraction
- Utilizes a deep neural network to classify intents
- Provides responses based on classified intents

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chatbot-deep-learning.git
   cd chatbot-deep-learning
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

## Data

- **intents.json**: Contains the different intents the chatbot can understand and respond to.
- **words.pkl** and **classes.pkl**: Pickled files containing the preprocessed vocabulary and intent classes.

## Training the Model

1. Preprocess the data:
   - Tokenize and lemmatize the sentences.
   - Create a bag-of-words model.
   - Save the processed words and classes.

2. Build and train the neural network model:
   - Define a Sequential model with Dense and Dropout layers.
   - Compile the model using SGD optimizer.
   - Train the model on the preprocessed data.

## Usage

1. Load the trained model and necessary data:
   ```python
   from keras.models import load_model
   import nltk
   import numpy as np
   import json
   import pickle

   model = load_model('chatbot_model.h5')
   intents = json.loads(open('intents.json').read())
   words = pickle.load(open('words.pkl', 'rb'))
   classes = pickle.load(open('classes.pkl', 'rb'))
   ```

2. Define helper functions to preprocess input and predict the intent:
   ```python
   def clean_up_sentence(sentence):
       sentence_words = nltk.word_tokenize(sentence)
       sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
       return sentence_words

   def bag_of_words(sentence, words, show_details=True):
       sentence_words = clean_up_sentence(sentence)
       bag = [0] * len(words)
       for s in sentence_words:
           for i, word in enumerate(words):
               if word == s:
                   bag[i] = 1
                   if show_details:
                       print("found in bag: %s" % word)
       return np.array(bag)

   def predict_class(sentence):
       p = bag_of_words(sentence, words, show_details=False)
       res = model.predict(np.array([p]))[0]
       ERROR_THRESHOLD = 0.25
       results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
       results.sort(key=lambda x: x[1], reverse=True)
       return_list = []
       for r in results:
           return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
       return return_list

   def getResponse(ints, intents_json):
       tag = ints[0]['intent']
       list_of_intents = intents_json['intents']
       for i in list_of_intents:
           if i['tag'] == tag:
               result = random.choice(i['responses'])
               break
       return result
   ```

3. Create a command-line interface to interact with the chatbot:
   ```python
   print("Start chatting with the bot (type 'quit' to stop)!")
   while True:
       message = input("You: ")
       if message.lower() == "quit":
           break

       ints = predict_class(message)
       res = getResponse(ints, intents)
       print("Bot:", res)
   ```

## Example

```
Start chatting with the bot (type 'quit' to stop)!
You: hi
Bot: Hi there, how can I help?
You: i am good how are you?
Bot: Good to see you again
You: yup what all can you do?
Bot: Offering support for Adverse drug reaction, Blood pressure, Hospitals and Pharmacies
You: tell me about blood pressure
Bot: Navigating to Blood Pressure module
You: what is the normal blood pressure?
Bot: Navigating to Blood Pressure module
You: quit
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [NLTK](https://www.nltk.org/)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)

Feel free to contribute to this project by submitting issues or pull requests.
