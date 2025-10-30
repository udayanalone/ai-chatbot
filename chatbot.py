
import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
nltk.download('popular', quiet=True)
nltk.download('punkt')
nltk.download('wordnet')

# Reading in the corpus
with open('chatbot.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# Tokenization
sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)  # converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    return None

def get_response(user_input):
    """Process user input and return bot's response"""
    user_input = user_input.lower()
    
    # Check for greetings
    greet_response = greeting(user_input)
    if greet_response is not None:
        return greet_response
    
    # Generate response using TF-IDF and cosine similarity
    robo_response = ''
    sent_tokens.append(user_input)
    
    try:
        TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx = vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        
        if req_tfidf == 0:
            robo_response = "I am sorry! I don't understand you. Could you rephrase that?"
        else:
            robo_response = sent_tokens[idx]
            
        # Remove the user input from the sentences
        if user_input in sent_tokens:
            sent_tokens.remove(user_input)
            
        return robo_response
        
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        return "I'm having trouble understanding. Could you try rephrasing that?"

# This part is kept for backward compatibility
def response(user_input):
    return get_response(user_input)
