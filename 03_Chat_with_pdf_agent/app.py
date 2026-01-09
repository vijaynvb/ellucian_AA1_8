import streamlit as st
from bedrock_agent import invoke_agent


st.title("Chat with PDF Agent using Bedrock")
st.write("Chat with uploaded PDF document and chat with it using Bedrock-powered agent.")
user_input = st.text_input("user: ","")
if st.button("Send"):
    if(user_input):
        with st.spinner("waiting for agent response..."):
            response = invoke_agent(user_input)
        st.success("Agent Response Recived")
        st.write(f"agent: {response}")
        #st.balloons()
    else:
        st.error("Please enter a message to send to the agent.")