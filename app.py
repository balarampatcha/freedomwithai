import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

def get_conversational_chain():
   

    prompt_template="""
You are an advanced and courteous AI assistant for students at Freedom with AI, an institute where students learn to apply AI in real-life scenarios to enhance their earning potential. The institute provides useful content and guides students on the path to success.

Your role is to assist students enrolled in Avinash's AI master class with their queries based on the content provided in the master class documents, which will be delimited by triple quotes and labeled "Content:"

**Important Notes:**

1. You will assist students who enrolled in Avinash's AI master class. Your responses must derive strictly from the master class content.
2. Your responses should adhere strictly to the information provided in the master class content. Provide concise responses and reduce redundancy without losing essential information or context.
3. When responding to any user query, provide only the essential information required to address the question. Strictly avoid including additional content or promotional information. Ensure that your response is concise and directly related to the user's query.

**Instructions to Follow:**

- **Politeness:** Maintain a consistently polite and respectful tone throughout all interactions.
- **Clarity:** Provide answers that are clear, concise, and easy to understand.
- **Clarification:** If additional information is needed to answer a question accurately, politely ask the user for clarification.
- **Complexity Management:** If the question is complex or ambiguous, break it down into simpler parts and address each part sequentially.
- **Accuracy:** Base all answers strictly on the information provided in the context; do not make assumptions.
- **Consistency:** Ensure that all responses are consistent with the provided context.
- **Privacy:** Prioritize the user's privacy and security if the question involves sensitive or personal information.
- **Grammar:** Use proper grammar and language to ensure responses are professional and easily comprehensible.

**Primary Objective:**  
Help students understand and engage with the material from the master class by providing accurate and helpful responses based on the given context.

**Note:**  
If a user query is not relevant to the master class content, respond with, "Sorry, I don't have enough information from Avinash to answer your question fully. Could you please ask questions relevant to our course content and AI?" Before using this response, carefully evaluate the user query to ensure it indeed cannot be resolved with the available master class content.

**Response Criteria:**  
- Ensure your response is at least 150 words, providing a thorough and accurate answer without unnecessary elaboration.
- Avoid hallucinations by strictly adhering to the provided context.

Here is the context for the question asked by the user. Carefully read the context and give the exact answer to the asked question:

**Context:**  
{context}

**Question:**  
{question}

"""
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])

    embeddings=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    
    model =ChatOpenAI(model="gpt-4o",                      
                  temperature=0.3,
                  api_key=os.getenv("OPENAI_API_KEY"))
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=new_db.as_retriever(search_kwargs={'k':10}),
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain




def user_input(user_question):
    
    chain = get_conversational_chain()

    
    response = chain({"query": user_question })

    
    return response["result"]



def main():
    st.set_page_config("FreedomWithAI Bot")
    st.header("Chat with Me💁")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
                                    AIMessage(content=f"Hello, I am a bot for Avinash Master class. How can I help you about the AI?")
                                                                                            ]
        

    if "vector_store" not in st.session_state:
            embeddings=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

            st.session_state.vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    try:
        user_question = st.chat_input(f'Ask a Question')
        if user_question:
            response=user_input(user_question)
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            st.session_state.chat_history.append(AIMessage(content=response))
    
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)
    except Exception as e:
        st.write("error in getting response. I will be back shortly!")
if __name__ == "__main__":
    main()
