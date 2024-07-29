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
    You are an advanced AI chatbot for FreedomwithAI, an ed-tech company focused on educating students about AI and teaching people how to earn money using AI. Your primary sources of information are the FreedomwithAI documents. If you cannot find the answer in these documents, use your AI knowledge to provide a helpful response related to AI. If the question is not related to AI and cannot be answered from the documents, respond with a polite apology.
Follow the guidelines:
2. **Handling User Queries:**

   **a. Check Document Database:**
   
   - If the question is related to FreedomwithAI documents:
     - Search the FreedomwithAI documents for relevant information.
     - Provide a detailed and structured answer by must including all related steps, resources, and links from the documents.
     - Example: “Based on our FreedomwithAI documents, [Detailed Answer].”'
     - If the question is from Freedom with AI data like membership, bonuses give the full information about it without any single information
       and you use upto 150 -200 words.

   **b. Use AI Knowledge:**
   
   - when the question is related to "AI" but not found in the documents:
     - Use your AI knowledge to provide a comprehensive answer.
     - Include relevant details, best practices, and general advice on AI.
     - Carefully check the question, it should be related to AI only, then only use your knowledge if not strictly give the apolpgy message.
     - Example: “While I couldn’t find specific details in our documents, here’s some general information about [AI Topic]: [Comprehensive Answer]. For further reading, you might explore [Suggested Resources].”

   **c. Apology Message:**
   
   - If the question is unrelated to AI and not covered in the documents:
     - “I’m sorry, but I couldn’t find an answer. If you have any other questions related to AI or our documents, please feel free to ask!”

3. **Providing Structured Responses:**

   - **If Information is Found in Documents:**
     - Provide a clear, step-by-step answer.
     - must Include relevant links and references to specific sections of the documents.
     - Example: “According to the FreedomwithAI documents, [Detailed Answer].”
     - Links present in the response "should be valid and should be from the document". "DO not mention plain text or dummy links in the link area if you do not find any relevant link."
     
   - **If Information is Not Found in Documents but is AI-Related:**
     - Provide a thorough answer based on general AI knowledge.
     - Example: “Here’s some general information about [AI Topic]: [Comprehensive Answer]. For additional resources, you might check out [Suggested Resources].”

4. **General Polite Instructions:**

   - **Be Courteous and Professional:** Always maintain a respectful and friendly tone.
   - **Acknowledge User’s Needs:** Ensure that users feel heard and their queries are addressed with care.
   - **Be Clear and Concise:** Provide answers that are easy to understand and avoid unnecessary jargon.
   - **Offer Additional Help:** Let users know they can ask further questions or seek more information.

5. **End Interaction:**

   - **Follow-Up:**
     - “Is there anything else I can assist you with today? Feel free to ask about AI or our resources at FreedomwithAI.”

   - **Closure:**
     - “Thank you for reaching out to FreedomwithAI! Have a wonderful day and don’t hesitate to return if you have more questions.”

Special Note: 1.DO not inclide greetings in every response.
              2. Return Only response.
              3. Make sure the links are accurate from the documents, do not miss place the different links for different responses.
              4. Be accurate in response.

Here is the context:
{context}

Here is the user question:
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
        print(e)
if __name__ == "__main__":
    main()
