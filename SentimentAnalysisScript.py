import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

U1F60A = '\U0001F60A'  # Assign the Unicode value for the emoji happy
U1F612 = '\U0001F612'  # Unamused face
U1F610 = '\U0001F610'  # Neutral face

def sentiment_finder(text):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    index = np.where(scores==scores.max())[0][0]   # return the index of maximum value among three scores determining the sentiments
    if (index==0): 
        return 'Negative', U1F612
    elif (index==1):
        return 'Neutral', U1F610
    else:
        return 'Positive',U1F60A
    
print(sentiment_finder('Hello, boy'))