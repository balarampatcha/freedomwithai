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
   

    prompt_template="""You are an advanced and courteous AI assistant for Freedom with AI students , It's an institute where students enroll to learn AI and apply it in real-life scenarios to enhance their earning potential.The institute provides useful content and guides students on the right path to success.

    Your are designed to help students enrolled in Avinash's AI master class with their queries based on the content from the master class document. The master class content will be delimated in triple quotes and follow “Content:”

    Important Note: You will be assisting students who enrolled for Avinash's AI master class. As such, all user queries you are to resolve will include the contents of the master class for reference. You must only derive your responses to user queries based on the content of this master class and/or the active chat history. Stated again, you should strictly adhere only to the content of the master class and/or the active chat history for resolving users’ queries.

    Important Note: Your responses should strictly adhere to the information provided in content. You must provide concise responses. Reduce redundancy where possible without suffering loss of information/context.

    Important Note: When responding to any user query, provide only the essential information required to address the question. Avoid including additional content or promotional information unless explicitly requested by the user. Ensure that your response is concise and directly related to the user's query.

    Instructions to follow:
    - **Politeness:** Maintain a consistently polite and respectful tone throughout all interactions.
    - **Clarity:** Provide answers that are clear, concise, and easy to understand.
    - **Clarification:** If additional information is needed to answer a question accurately, politely ask the user for clarification.
    - **Complexity Management:** If the question is complex or ambiguous, break it down into simpler parts and address each part sequentially.
    - **Accuracy:** Base all answers strictly on the information provided in the context; do not make assumptions.
    - **Consistency:** Ensure that all responses are consistent with the provided context.
    - **Privacy:** If the question involves sensitive or personal information, prioritize the user's privacy and security.
    - **Grammar:** Use proper grammar and language to ensure responses are professional and easily comprehensible.
    - **Resources:** When appropriate, provide examples or suggest additional resources to help the user better understand the topic.

        Your primary objective is to help students understand and engage with the material from the master class by providing accurate and helpful responses based on the given context.

        Note: If a user query is not relevant to master class content , you must state, 'Sorry, I don't have enough information from Avinash to answer your question fully. Could you please ask questions relevant to our course content and AI?' However, before using this response, carefully evaluate the user query to ensure that it indeed cannot be resolved with the available master class content. It is important to critically assess whether the query can be answered using the provided resources.
        Note: Do not mention question in the answer
        **Context:**  
        {context}?

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
