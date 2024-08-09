import os
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()


def load_files_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):  # Assuming text files
            file_path = os.path.join(directory_path, filename)
            print(f"Loading file: {file_path}")
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
    return documents


def update_faiss_index(
    index_path: str,
    new_index_path: str,
    text_file_path: str,
    api_key: str,
    chunk_size: int = 4000,
    chunk_overlap: int = 500
):
    # Initialize the embeddings model
    embeddings = OpenAIEmbeddings(api_key=api_key)

    # Load the existing FAISS index
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    
    # Load new documents
    pages = load_files_from_directory(text_file_path)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)


    # Add new embeddings to the vectorstore
    vectorstore.add_documents(docs)

    # Save the updated FAISS index
    vectorstore.save_local(new_index_path)

# Example usage:
update_faiss_index(
    index_path='faiss_index',
    new_index_path='faiss_index',
    text_file_path='data/',
    api_key=os.getenv("OPENAI_API_KEY")
    )
