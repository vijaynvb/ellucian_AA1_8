
import streamlit as st

prompt = st.chat_input("Say something")
if prompt:
    with st.chat_message("User"):
      st.write(f"User has sent the following prompt: {prompt}")
            