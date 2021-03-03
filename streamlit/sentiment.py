# imports
import streamlit as st
import pandas as pd
from afinn import Afinn

st.title('Test your data!')


st.markdown("You could imagine that we could use a very simple tool, like below, to validate input from models and other external datasets")

# Check your customer's statement here:
# user_input = st.text_area("label goes here", default_value_goes_here)
user_input = st.text_area("What is the text you want to verify")

afinn = Afinn(language='en')


if st.button('Submit'):
    score = afinn.score(user_input)
    st.markdown("----")
    st.write('Sentiment Score (afinn):  %s' % score)
    
