import base64
import pickle
import string
import warnings

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from stocktwitsapi import stocktwits_scrap
from sklearn.preprocessing import MinMaxScaler

lemmatizer = WordNetLemmatizer()
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))

key = "961705e28c14170672d45212990087314a455244"
warnings.filterwarnings("ignore")

word_to_indx = open('word_to_indx', "r", encoding='UTF-8').readlines()


def remove_punctuation(data):
    out_ = []
    for tweet in data:
        txt = tweet.lower()
        txt_punch_removed = ''.join([char for char in txt if char not in string.punctuation])
        sent = [word for word in txt_punch_removed.split(' ') if word not in stop_words]
        final = [lemmatizer.lemmatize(word) for word in sent]
        out = []
        for word in final:
            if word in word_to_indx:
                out.append(word_to_indx[word])
            else:
                out.append(0)
        while len(out) != 50:
            out.append(0)
        out_.append(out)
    return np.array(out_)


def set_bg_hack(main_bg):
    """
    A function to unpack an image from root folder and set as bg.

    Returns
    -------
    The background.
    """
    # set bg name
    main_bg_ext = "png"

    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    body {{
    background-image: url("data:image/png;base64,{bin_str}");
    background-size: cover;
    font-color: black;
    }}
    </style>
    '''

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


set_bg_hack('background.jpg')
st.markdown(f'<h1 style="color:white;">{"Stock Sentiment Analysis"}</h1>', unsafe_allow_html=True)
with st.form("my_form"):
    textSymbol = st.text_input("Enter your symbol")
    # option = st.selectbox('How would you like to be contacted?',('Email', 'Home phone', 'Mobile phone'))
    checkbox_val = st.checkbox("Also get the time series")

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        # st.write("Symbol Entered", textSymbol, "checkbox", checkbox_val, "Option selected", option)

        tickerList = [textSymbol]
        for ticker in tickerList:
            filename = ticker.lower()
            df_stocktwits, last_id = stocktwits_scrap(ticker=ticker)
            df_stocktwits.to_csv('%s_api_data.csv' % filename, index=False)
            for i in range(10):
                df_stocktwits, last_id = stocktwits_scrap(ticker=ticker, base=last_id)
                if last_id != 0:
                    df_stocktwits.to_csv('%s_api_data.csv' % filename, mode='a', index=False, header=False)
                    print("Scraped: " + str(i * 30) + " twits")
        st.dataframe(df_stocktwits)
        only_data = df_stocktwits['body']
        only_data.drop_duplicates(inplace=True)
        only_data.dropna(axis=0, inplace=True)
        new_cleaned_data = remove_punctuation(only_data)
        # model1 = pickle.load(open('model.pkl', 'rb'))
        # model1 = joblib.load('model.sav')
        model1 = tf.keras.models.load_model('model.h5')

        ####### new one starts here

        # predict the sentiment on test
        def pos_neg(y_pred):
            dec = ''
            if np.argmax(y_pred) == 0:
                dec = 'negative'
            elif np.argmax(y_pred) == 1:
                dec = 'neutral'
            else:
                dec = 'positive'
            return dec


        def predict(data):
            y_pred = model1.predict(data)
            dec = pos_neg(y_pred)
            return dec

        print(predict(new_cleaned_data.reshape(-1, 50)))

        if checkbox_val:
            df = pdr.get_data_tiingo(textSymbol, api_key=key)
            date = df.reset_index()['date'].dt.date
            df1 = df.reset_index()['close']
            fig = plt.figure(figsize=(5, 5))

            plt.plot(date, df1)
            plt.xticks(rotation=90)
            st.pyplot(fig)
            scaler = MinMaxScaler(feature_range=(0, 1))
            df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
            model = pickle.load(open('timeseries_model.pkl', 'rb'))
            x_input = df1[len(df1) - 100:].reshape(1, -1)

            temp_input = list(x_input)
            temp_input = temp_input[0].tolist()

            lst_output = []
            n_steps = 100
            i = 0
            while i < 30:

                if len(temp_input) > 100:
                    # print(temp_input)
                    x_input = np.array(temp_input[1:])
                    # print("{} day input {}".format(i,x_input))
                    x_input = x_input.reshape(1, -1)
                    x_input = x_input.reshape((1, n_steps, 1))
                    # print(x_input)
                    yhat = model.predict(x_input, verbose=0)
                    # print("{} day output {}".format(i,yhat))
                    temp_input.extend(yhat[0].tolist())
                    temp_input = temp_input[1:]
                    # print(temp_input)
                    lst_output.extend(yhat.tolist())
                    i = i + 1
                else:
                    x_input = x_input.reshape((1, n_steps, 1))
                    yhat = model.predict(x_input, verbose=0)
                    temp_input.extend(yhat[0].tolist())
                    lst_output.extend(yhat.tolist())
                    i = i + 1

            day_new = date[(len(date) - 130): (len(date) - 30)]
            day_pred = date[(len(date) - 30): len(date)]
            fig = plt.figure(figsize=(5, 5))
            fig, (ax2, ax1) = plt.subplots(1, 2)
            fig.suptitle('Comparision of the original and predicted trend')

            ax1.plot(day_new, scaler.inverse_transform(df1[len(df1) - 100:]))
            ax1.set_xticklabels(day_new, rotation='vertical')
            ax1.plot(day_pred, scaler.inverse_transform(lst_output))
            ax1.set_xticklabels(day_pred, rotation='vertical')
            ax1.title.set_text('predicted Trend')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')

            df3 = df1.tolist()
            df3.extend(lst_output)

            ax2.plot(date[len(date) - 130:], scaler.inverse_transform(df3[len(df3) - 130:]))
            ax2.set_xticklabels(date[len(date) - 130:], rotation='vertical')
            ax2.title.set_text('original Trend')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Price')

            st.pyplot(fig)

    # run the sentiment analysis here and we are done
