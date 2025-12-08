from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai   import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("mediumblog1.txt")
    documents = loader.load()
    print(documents)

    print("Documents loaded successfully")
    # chunk size is heuristic, it is the size of the chunk you want to create and understand the context
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)
    print(f"Number of chunks: {len(chunks)}")

    # embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small")
    # text-embedding-ada-002 is default model for openai embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small", dimensions=512)

    print("Embeddings created successfully")
    PineconeVectorStore.from_documents(documents=chunks, embedding=embeddings, index_name=os.getenv("PINECONE_INDEX"))
    
    print("Documents added to Pinecone successfully")