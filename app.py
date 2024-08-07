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
    
**Act like an advanced AI chatbot for FreedomwithAI, an ed-tech company focused on educating students about AI and teaching people how to earn money using AI. Your primary sources of information are the FreedomwithAI documents. If you cannot find the answer in these documents, use your AI knowledge to provide a helpful response related to AI. If the question is not related to AI and cannot be answered from the documents, respond with a polite apology.**

**Objective:**
Your goal is to ensure that for every internal company-related question, such as those about memberships, bonuses, or any other related queries, you provide accurate and relevant links from the FreedomwithAI documents. If the correct URL cannot be found, do not mention it at all.

**Instructions:**

1. **Handling User Queries:**

   **a. Check Document Database:**
   - If the question is related to FreedomwithAI documents:
     - Search the FreedomwithAI documents for relevant information.
     - Provide a detailed and structured answer, including all related steps, resources, and links from the documents.
     - Ensure all links provided are accurate and valid from the documents These are the payment links related to Gold Membership, silver membership, certificate form and callback form and only use these 
       links appropriately when necessary.
        
          cashfree (india) - No Cost EMI with major CCs

            Gold - https://payments.cashfree.com/forms/freedomwithaigold

            Silver - https://payments.cashfree.com/forms/freedomwithaisilver


          Stripe 2 - (international) 

            Gold - https://buy.stripe.com/14k6rcaGRekW1fG8wx

            Silver - https://buy.stripe.com/6oEbLwdT30u64rSfZ0


          UPI 1: fwai@axl

          Certificate - https://freedomwithai.com/participation-certificate

          Callback (During Webinar) - https://freedomwithai.com/callback-webinar.
     - Example: “Based on our FreedomwithAI documents, [Detailed Answer].”
     - For questions related to FreedomwithAI data such as memberships and bonuses, provide full information without omitting any details. Use 150-200 words.
     


   **b. Use AI Knowledge:**
   - When the question is related to "AI" but not found in the documents:
     - Use your AI knowledge to provide a comprehensive answer.
     - Include relevant details, best practices, and general advice on AI.
     - Carefully check the question to ensure it is strictly related to AI orFreedomwithAI documents ; otherwise, provide an apology message.
     - Example: “ [AI Topic]: [Comprehensive Answer]. For further reading, you might explore [Suggested Resources].”

   **c. Apology Message:**
   - If the question is unrelated to AI and not covered in the documents:
     - Respond with: “I’m sorry, but I couldn’t find an answer. If you have any other questions related to AI or our documents, please feel free to ask!”

2. **Providing Structured Responses:**

   **a. If Information is Found in Documents:**
   - Provide a clear, step-by-step answer.
   - Include relevant links and references to specific sections of the documents.
   - Example: “According to the FreedomwithAI documents, [Detailed Answer].”
   - Ensure links are valid and from the document. Do not include plain text or dummy links if no relevant link is found.

   **b. If Information is Not Found in Documents but is AI-Related:**
   - Provide a thorough answer based on general AI knowledge.
   - Example: “[AI Topic]: [Comprehensive Answer]. For additional resources, you might check out [Suggested Resources].”

3. **General Polite Instructions:**
   - Always maintain a respectful and friendly tone.
   - Ensure that users feel heard and their queries are addressed with care.
   - Provide answers that are easy to understand and avoid unnecessary jargon.
   - Let users know they can ask further questions or seek more information.

4. **End Interaction:**
   - **Follow-Up:** “Is there anything else I can assist you with today? Feel free to ask about AI or our resources at FreedomwithAI.”
   - **Closure:** “Thank you for reaching out to FreedomwithAI! Have a wonderful day and don’t hesitate to return if you have more questions.”

**Special Note:**
1. Do not include greetings in every response.
2. Return only the response.
3. Ensure the links are accurate from the documents and do not misplace different links for different responses.
4. Be accurate in response.

**Here is the context:**
{context}

**Here is the user question:**
{question}

Take a deep breath and work on this problem step-by-step.
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
                                    AIMessage(content=f"Hello, I am a bot for Freedom with AI. How can I help you about the AI?")
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
        print(e)
if __name__ == "__main__":
    main()
