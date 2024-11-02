import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


stemmer = PorterStemmer()

def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) 
    text = re.sub(r'#[A-Za-z0-9]+', '', text)  
    text = re.sub(r'http\S+', '', text)       
    text = re.sub(r'[0-9]+', '', text)         
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))  
    text = text.strip(' ')  
    return text

def casefoldingText(text):
    return text.lower()  

def tokenizingText(text):
    return word_tokenize(text)

def filteringText(text):
    listStopwords = set(stopwords.words('english'))
    filtered = [word for word in text if word not in listStopwords]
    return filtered

def stemmingText(text):
    stemmed_words = [stemmer.stem(word) for word in text]
    return ' '.join(stemmed_words)

def preprocess_text(text):
    text = cleaningText(text)
    text = casefoldingText(text)
    tokens = tokenizingText(text)
    filtered_tokens = filteringText(tokens)
    stemmed_text = stemmingText(filtered_tokens)
    return stemmed_text
