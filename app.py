import streamlit as st
import pandas as pd
import pickle
# from MMS_Project import X_test, Y_test
from stemming import stemming

def btn_sbmt(author, title):
    news = author+title
    data = pd.DataFrame({'content':[news]})
    data['content'] = data['content'].apply(stemming)
    
    vectorizer_model = pickle.load(open('vectorizer_model.sav','rb'))
    data_v = vectorizer_model.transform(data['content'].values)

    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    result = loaded_model.predict(data_v)

    print(data['content'])
    print("result:",result)
    print('------------')
    FinalResult = ''
    if result[0] == 0:
        FinalResult = '<p style="font-family:sans-serif; color:Green; font-size: 100px; text-align:center">Real News</p>'.format(FinalResult)
    else:
        FinalResult = '<p style="font-family:sans-serif; color:Red; font-size: 100px; text-align:center">Fake News</p>'.format(FinalResult)
   
    st.markdown(FinalResult, unsafe_allow_html=True)
        

# st.markdown("""
#     <style>
#     .reportview-container {
#         background: url("https://media.istockphoto.com/photos/abstract-digital-news-concept-picture-id1290904409")
#     }
#     .css-1mg5w7t-{
#         background:white
#     }
#     </style>
#     """,unsafe_allow_html=True)

st.title("Fake News Detection")
st.write('Insert News Details')
form = st.form("my_form", clear_on_submit=False)
author = form.text_input(label="Author")
title = form.text_input(label="Title")
text = form.text_area(label="Text")
form.form_submit_button(label="Submit", on_click=btn_sbmt(author, title))
