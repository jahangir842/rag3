import os
from dotenv import load_dotenv
import chromadb
import requests
from chromadb.utils import embedding_functions
import PyPDF2

# Load environment variables from .env file
load_dotenv()

# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"

# Initialize embedding function using ChromaDB's built-in support
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='all-MiniLM-L6-v2'
)

collection = chroma_client.get_or_create_collection(
    name=collection_name, 
    embedding_function=sentence_transformer_ef
)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            text_content = extract_text_from_pdf(file_path)
            documents.append({"id": filename, "text": text_content})
    return documents


# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Ensure we don't split in the middle of a word or line
        if end < len(text):
            # Try to find the next whitespace
            while end < len(text) and not text[end].isspace():
                end += 1
        chunks.append(text[start:end].strip())
        start = end - chunk_overlap
    return [chunk for chunk in chunks if chunk]  # Remove empty chunks

# Load documents from the directory
directory_path = "./cv"  # Changed from news_articles to cv
documents = load_documents_from_directory(directory_path)

print(f"Loaded {len(documents)} documents")
# Split documents into chunks
chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print(f"==== Splitting {doc['id']} into chunks ====")
    for i, chunk in enumerate(chunks):
        if chunk.strip():  # Only add non-empty chunks
            chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

# Generate embeddings and insert into Chroma
for doc in chunked_documents:
    print(f"==== Generating embeddings for {doc['id']} ====")
    collection.upsert(
        ids=[doc["id"]], 
        documents=[doc["text"]]
    )


# Function to query documents
def query_documents(question, n_results=3):  # Increased n_results for better context
    results = collection.query(query_texts=question, n_results=n_results)
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("\n==== Relevant chunks used for answering ====")
    for i, chunk in enumerate(relevant_chunks, 1):
        print(f"\nChunk {i}:")
        print("-" * 80)
        print(chunk)
        print("-" * 80)
    print("\n==== Generating answer based on above chunks ====")
    return relevant_chunks


# Function to generate a response using local llama.cpp server
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an AI assistant analyzing resumes and CVs. Use the following pieces of "
        "retrieved context to answer the question about the candidates. If you don't know "
        "the answer, say that you don't know. Keep the answer concise and professional."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = requests.post(
        "http://localhost:8000/completion",
        json={
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 250  # Increased for more detailed responses about resumes
        }
    )
    
    return response.json()["content"]


# Example usage
if __name__ == "__main__":
    question = "What are the skills and qualifications Pakeeza?"
    print("==== Asking question ====") 
    print(question)         
    relevant_chunks = query_documents(question)
    answer = generate_response(question, relevant_chunks)
    print(answer)
