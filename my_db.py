import os
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader


def get_text_chunks(text_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=200)
    documents = text_splitter.split_documents(text_documents)
    return documents


def get_vector_store(documents):
    embeddings =  OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vector_store = FAISS.from_documents(documents,embeddings)
    vector_store.save_local("faiss_index")

    
loader=TextLoader("main_document.txt")

text_documents=loader.load()

chunks= get_text_chunks(text_documents)

get_vector_store(chunks)