import streamlit as st
# import cv2
import pandas as pd
import numpy as np
# from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
# import my_db
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))




def get_conversational_chain():

    prompt_template = """
    Being a excellent AI helper,
    Answer the question as detailed as possible from the provided context,
    Context:\n {context}?\n
    Question: \n{question}\n
    and other scenario is, if there is no exact answer for the question, please answer the question as "Sorry I dont have enough information from Avinash".
    and most important thing is while answering the answer, please answer as you are giving the answer like I can answer your question, Sorry I dont 
    have enough information on this.
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    # st.write("Reply: ", response["output_text"])
    return response["output_text"]



def main():
    st.set_page_config("FreedomWithAI Bot")
    st.header("Chat with Me💁")
#     image='''
# '''
#     st.markdown(image,unsafe_allow_html=True)
    # # user_question = st.text_input("Ask a Question")
    # st.write('Please upload the QR code')
    # # if user_question:
    # #     user_input(user_question)

    # with st.sidebar:
    #     st.title("Menu:")
    #     st.header("Settings")
    # # website_url = st.text_input("Website URL")
    #     uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    #     if uploaded_image:
    #         with st.spinner("Processing..."):
    #             product,raw_text = get_QR_text(uploaded_image)
    #             text_chunks = get_text_chunks(raw_text)
    if "chat_history" not in st.session_state:
        # response=user_input('Kindly brief the outlines of the master class and If possible please keep those important topics in bullet points')
        st.session_state.chat_history = [
                                    AIMessage(content=f"Hello, I am a bot for Avinash Master class. How can I help you about the AI?"),
                                    # AIMesssage(content=response)
                                                                                            ]
        
        # st.session_state.chat_history.append()


    if "vector_store" not in st.session_state:
            embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
            st.session_state.vector_store = FAISS.load_local("faiss_index", embeddings)

    # get_vector_store(text_chunks)
    # st.success("Done")
    try:
        user_question = st.chat_input(f'Ask a Question')
        if user_question:
            response=user_input(user_question)
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            st.session_state.chat_history.append(AIMessage(content=response))
    
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    # avatar_url = 'https://www.bing.com/images/search?view=detailV2&ccid=IFL70ddS&id=3A976BA130023A9A8D287083D1CA7E8BE9430F22&thid=OIP.IFL70ddSbrxHiHBNMEJ3jAHaFj&mediaurl=https%3a%2f%2fwww.vhv.rs%2ffile%2fmax%2f20%2f201053_google-logo-png.png&exph=1200&expw=1600&q=google&simid=607990172418264597&FORM=IRPRST&ck=7A3D084E848ACF5B44BDAD657547BE8C&selectedIndex=1&itb=0'
                    # st.image(avatar_url, width=40)
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)
    except Exception as e:
        st.write(e)
if __name__ == "__main__":
    main()
