import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from nltk.stem.porter import PorterStemmer
import os


nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()


def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  y =[]
  for i in text:
    if i.isalnum():
      y.append(i)
  text = y[:] #mutable list so cloning not copy directly
  y.clear()
  for i in text:
    if i not in stopwords.words("english") and i not in string.punctuation:
      y.append(i)
  text = y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))
  return " ".join(y)

with open("require.css") as source_des:
    st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)

tfidf = pickle.load(open("vectorizer.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))
st.title("Text Classifier")

st.markdown("<p style='font-size: 18px;'>Enter the message</p>", unsafe_allow_html=True)
input_sms = st.text_area("")

# Prediction button
if st.button("Predict"):
    if not input_sms.strip():
        st.markdown("<p style='font-size: 17px; color: Red;'>Please enter some text.</p>", unsafe_allow_html=True)
    else:
        # Preprocess text
        transformed_sms = transform_text(input_sms)
        # Vectorize text
        vector_input = tfidf.transform([transformed_sms])
        # Predict
        result = model.predict(vector_input)[0]
        # Display result
        if result == 1:
            st.header("Positive")
        elif result == 0:
            st.header("Neutral")
        else:
            st.header("Negative")