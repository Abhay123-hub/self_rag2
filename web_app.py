## creating the web application for my project
import streamlit as st
from transformers import pipeline
from chatbot import chatbot
import uuid
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
mybot = chatbot()
workflow = mybot() ##

## set up the streamlit UI(user interface)
st.title("ChatBot With LangGraph 📚 😄 🤖 🧐 ")
st.write("Ask any question I will try to answer it")

## input text box for the question
question = st.text_input("Enter your question here 🤖")


inputs = {"question": question}

## button to get the response from my chatbot

if st.button("Get Answer"):
    if input:
        response = workflow.invoke(inputs,config = config)
        st.write("**Answer**",response["generation"]) 