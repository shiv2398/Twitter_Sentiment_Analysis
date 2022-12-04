import streamlit as st
import pickle

model,cv=pickle.load(open('/home/sahitya/visual_studio/Twitter Sentiment Analysis/best_model.pkl','rb'))

st.title('Twitter Sentiment Analysis')
input_text=st.text_input('Tweet Here !')
def predict():
    input_array=cv.transform(list(input_text))

    pred=model.predict(input_array)[0]

    if pred==4:
        st.success('Tweet is positive :thumbsup:')
    else:
        st.error('Tweet is Negative :thumbdown:')

st.button('Predict',on_click=predict)



