import streamlit as st

st.set_page_config(page_title="Ai Voice Cloner", page_icon="🧠")
st.title("AI Voice Cloner")

input = st.text_input("Enter Your Input Text")

submit=st.button('Submit')