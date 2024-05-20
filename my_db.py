from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

df=pd.read_excel('record.xlsx')

df['text']=df['text'].apply(lambda x: x.replace('\n',' '))
df['text']=df['text'].apply(lambda x: x.replace('\t',' '))
df['text']=df['text'].apply(lambda x: x.replace('\r',' '))

chunks= get_text_chunks(df['text'].to_list()[0])

get_vector_store(chunks)