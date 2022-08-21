

################################################################################################
# Importing libraries
################################################################################################

import numpy as np
import math
import pandas as pd
import re
import sys
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from sklearn.utils import resample
import warnings
import seaborn as sns 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding,BatchNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
sns.set()
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import pickle



################################################################################################
# Defining functions
################################################################################################     
def remove_punctuation(tweet):
    tweet2=[char for char in tweet if char not in string.punctuation ]
    
    return ''.join(tweet2)


def create_index(x:list):
    
    wrd_idx = {}
    idx_wrd = {}
    idx_cnt = 1

    for sent in x:
        for word in sent:
            if word not in wrd_idx.keys():
                wrd_idx[word] = idx_cnt
                idx_wrd[idx_cnt] = word
                idx_cnt +=1
    return wrd_idx, idx_wrd

################################################################################################
# Main function
################################################################################################ 

def main(args = []):

    data=pd.read_csv("tweet_sentiment.csv")
    
    #clenaing data
    data.drop_duplicates(inplace=True)
    data.dropna(axis = 0, inplace = True)
    
    data['cleaned_tweets']=data['cleaned_tweets'].str.lower()

    data["cleaned_tweets"] = data["cleaned_tweets"].apply(remove_punctuation)

    stop_words=set(stopwords.words('english'))
    data['cleaned_tweets']=data['cleaned_tweets'].apply(lambda sent: [word for word in word_tokenize(sent) if word not in stop_words])

    # ps=PorterStemmer()
    # data['new_clean_tweets'] = data['cleaned_tweets'].apply(lambda words: [ps.stem(word) for word in words])

    # data['cleaned_tweets']=data['cleaned_tweets'].str.lower()
    # # data["cleaned_tweets"] = data["cleaned_tweets"].apply(remove_punctuation)
    # stop_words=set(stopwords.words('english'))

    # data['cleaned_tweets']=data['cleaned_tweets'].apply(lambda sent: [word for word in word_tokenize(sent) if word not in stop_words])

    ps=PorterStemmer()
    data['new_clean_tweets'] = data['cleaned_tweets'].apply(lambda words: [ps.stem(word) for word in words])

    #feature generation
    wrd_idx, idx_wrd = create_index(data["new_clean_tweets"])

    voc = len(wrd_idx.keys())
    voc += 1

    data["cleaned_tweets_feature"] = data["new_clean_tweets"].apply(lambda words: [wrd_idx[word] for word in words])

    #data preparation
    input_size = 50

    tweets_data = pad_sequences(data["cleaned_tweets_feature"], maxlen=input_size ,dtype="object", padding="post", truncating="post")
    tweets_data = tweets_data.astype(np.float32)

    sentiment_data = pd.get_dummies(data["sentiment"],drop_first=False).values
    sentiment_data = sentiment_data.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(tweets_data, sentiment_data, test_size=0.2)

    #building model
    model = Sequential()
    model.add(Embedding(voc, 32, input_length=input_size))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(2048,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    h = model.fit(X_train,y_train, validation_split=0.2, epochs=10, batch_size=32)

    print("Evaluate on test data")
    results = model.evaluate(X_test, y_test, batch_size=128)
    print("test loss, test acc:", results)
    pickle.dump(model, open('model.pkl', 'wb'))


if __name__ == "__main__":
    main(sys.argv)  