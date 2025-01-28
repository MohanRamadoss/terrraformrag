import requests
from bs4 import BeautifulSoup
import ollama
import os
import logging
import chromadb
from chromadb.utils import embedding_functions
import hashlib
from ollama_utils import (
    get_available_models,
    generate_code_ollama,
    ModelNotAvailableError,
    CodeGenerationError,
    OLLAMA_API_URL
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize global variables
CHUNKS = []

logger.info(f"RAG utils using Ollama API URL: {OLLAMA_API_URL}")

# Website URLs to scrape
PROVIDER_URL = "https://registry.terraform.io/browse/providers"
MODULE_URL = "https://registry.terraform.io/browse/modules"

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)

# Define recommended models for Terraform code generation
RECOMMENDED_MODELS = [
    "deepseek-r1:14b",
    "granite-code:20b",
    "deepseek-coder:33b",
    "qwen2.5-coder:14b",
    "codellama:13b"
]

def initialize_chroma_collection():
    """Initialize or get the ChromaDB collection for Terraform content."""
    try:
        collection = chroma_client.get_or_create_collection(
            name="terraform_docs",
            embedding_function=embedding_function
        )
        return collection
    except Exception as e:
        print(f"Error initializing ChromaDB collection: {e}")
        return None

def update_chroma_db(chunks):
    """Update ChromaDB with the latest chunks."""
    collection = initialize_chroma_collection()
    if not collection:
        return
    
    try:
        # Get existing IDs
        existing_ids = collection.get()['ids']
        if existing_ids:
            # Delete existing documents one by one
            for doc_id in existing_ids:
                collection.delete(ids=[doc_id])
        
        # Add new chunks
        if chunks:
            ids = [hashlib.md5(chunk.encode()).hexdigest() for chunk in chunks]
            collection.add(
                documents=chunks,
                ids=ids
            )
            print(f"Added {len(chunks)} documents to ChromaDB")
    except Exception as e:
        print(f"Error updating ChromaDB: {e}")

def semantic_search(query, n_results=5):
    """Perform semantic search using ChromaDB."""
    collection = initialize_chroma_collection()
    if not collection:
        return []
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, len(collection.get()['ids']))
        )
        return results['documents'][0] if results['documents'] else []
    except Exception as e:
        print(f"Error performing semantic search: {e}")
        return []

def scrape_website(url):
    """Scrapes text content from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text_content = ""
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
            text_content += element.text + "\n"
        return text_content
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None

def process_data(text_content):
    """Processes and chunks the scraped text."""
    if not text_content:
        return []
    text_content = " ".join(text_content.split()).strip()
    chunks = text_content.split("\n\n")
    if not chunks or len(chunks) < 2:
        chunks = text_content.split("\n")
    return chunks

def retrieve_relevant_chunks(query, chunks):
    """Basic keyword-based retrieval."""
    relevant_chunks = []
    query_lower = query.lower()
    for chunk in chunks:
        if query_lower in chunk.lower():
            relevant_chunks.append(chunk)
    return relevant_chunks

def initialize_data():
    """Initialize data by scraping and processing website content."""
    global CHUNKS
    
    try:
        print("Scraping Terraform Registry websites...")
        provider_content = scrape_website(PROVIDER_URL)
        module_content = scrape_website(MODULE_URL)
        ALL_CONTENT = (provider_content or "") + "\n\n" + (module_content or "")
        CHUNKS = process_data(ALL_CONTENT)
        
        if CHUNKS:
            print("Initializing ChromaDB with content...")
            update_chroma_db(CHUNKS)
            print(f"Processed {len(CHUNKS)} chunks of content")
        else:
            print("Warning: No content was processed")
        
        return CHUNKS
    except Exception as e:
        logger.error(f"Error initializing data: {e}")
        return []

# Initialize data when module is imported
CHUNKS = initialize_data()

# Ensure CHUNKS is available for import
__all__ = ['retrieve_relevant_chunks', 'generate_code_ollama', 'CHUNKS', 
           'semantic_search', 'get_available_models']

# Initialize available models
AVAILABLE_MODELS = get_available_models()
logger.info(f"Available models: {AVAILABLE_MODELS}")
