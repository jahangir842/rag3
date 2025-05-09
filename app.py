import os
from dotenv import load_dotenv
import chromadb
import requests
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import json

# Load environment variables from .env file
load_dotenv()

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"

# Custom embedding function using Sentence Transformers
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='all-MiniLM-L6-v2'
)

collection = chroma_client.get_or_create_collection(
    name=collection_name, 
    embedding_function=sentence_transformer_ef
)

# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents


# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


# Load documents from the directory
directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)

print(f"Loaded {len(documents)} documents")
# Split documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print("==== Splitting docs into chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

# Generate embeddings and insert into Chroma
for doc in chunked_documents:
    print("==== Generating embeddings... ====")
    collection.upsert(
        ids=[doc["id"]], 
        documents=[doc["text"]]
    )


# Function to query documents
def query_documents(question, n_results=2):
    results = collection.query(query_texts=question, n_results=n_results)
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks


# Function to generate a response using local llama.cpp server
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = requests.post(
        "http://localhost:8000/completion",
        json={
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 150
        }
    )
    
    return response.json()["content"]


# Example usage
if __name__ == "__main__":
    question = "Who is Jahangir Alam"
    relevant_chunks = query_documents(question)
    answer = generate_response(question, relevant_chunks)
    print(answer)
