from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import os
from dotenv import load_dotenv
import docx2txt

load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# df=pd.read_excel('data_trail.xlsx')

import pandas as pd

my_text = docx2txt.process("masterclass2.docx")

# df['text']=df['text'].apply(lambda x: x.replace('\n',' '))
# df['text']=df['text'].apply(lambda x: x.replace('\t',' '))
# df['text']=df['text'].apply(lambda x: x.replace('\r',' '))

chunks= get_text_chunks(my_text)

get_vector_store(chunks)