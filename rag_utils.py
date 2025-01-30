import requests
from bs4 import BeautifulSoup
import ollama
import os
import logging
import chromadb
from chromadb.utils import embedding_functions
import hashlib
from typing import List, Optional, Dict, Any  # Add typing imports
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

# Update URL constants
PROVIDER_URLS = [
    "https://registry.terraform.io/providers/hashicorp/aws/latest/docs",
    "https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs",
    "https://registry.terraform.io/providers/hashicorp/google/latest/docs",
    "https://registry.terraform.io/providers/hashicorp/kubernetes/latest/docs",
    "https://registry.terraform.io/providers/hashicorp/docker/latest/docs"
]

MODULE_URLS = [
    "https://registry.terraform.io/modules/terraform-aws-modules/vpc/aws/latest",
    "https://registry.terraform.io/modules/Azure/compute/azurerm/latest",
    "https://registry.terraform.io/modules/terraform-google-modules/network/google/latest",
    "https://registry.terraform.io/modules/terraform-aws-modules/eks/aws/latest",
    "https://registry.terraform.io/modules/terraform-aws-modules/rds/aws/latest"
]

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

def semantic_search(query, n_results=10):
    """Improved semantic search with better relevance filtering."""
    collection = initialize_chroma_collection()
    if not collection:
        logger.warning("ChromaDB collection not available, falling back to keyword search")
        return retrieve_relevant_chunks(query, CHUNKS, n_results)
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results * 2, len(collection.get()['ids']))  # Get more results for filtering
        )
        
        if results and 'documents' in results and results['documents']:
            documents = results['documents'][0]
            # Enhanced relevance filtering
            filtered_docs = []
            for doc in documents:
                # Check document relevance
                if len(doc.strip()) > 50:  # Minimum length check
                    relevance_score = sum(term.lower() in doc.lower() 
                                        for term in query.lower().split())
                    if relevance_score > 0:
                        filtered_docs.append((doc, relevance_score))
            
            # Sort by relevance and take top results
            filtered_docs.sort(key=lambda x: x[1], reverse=True)
            final_results = [doc for doc, _ in filtered_docs[:n_results]]
            
            logger.info(f"Semantic search found {len(final_results)} relevant results")
            return final_results
            
        logger.warning("No semantic search results found")
        return retrieve_relevant_chunks(query, CHUNKS, n_results)
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return retrieve_relevant_chunks(query, CHUNKS, n_results)

def retrieve_relevant_chunks(query: str, chunks: List[str], max_results: int = 10) -> List[str]:
    """Enhanced keyword-based retrieval with improved matching."""
    if not chunks:
        logger.warning("No chunks available for search")
        return []
        
    # Normalize query and create search terms
    query_terms = set(query.lower().split())
    search_terms = set()
    
    # Add variations of common terms
    for term in query_terms:
        search_terms.add(term)
        # Add AWS variations
        if term == 'aws':
            search_terms.update(['amazon', 'cloudwatch', 'ec2', 's3', 'rds'])
        # Add Azure variations
        elif term == 'azure':
            search_terms.update(['microsoft', 'azurerm', 'app'])
        # Add common Terraform terms
        elif term in ['create', 'setup', 'configure']:
            search_terms.update(['resource', 'module', 'provider'])
            
    logger.info(f"Search terms: {search_terms}")
    scored_chunks = []
    
    for chunk in chunks:
        chunk_lower = chunk.lower()
        # Calculate relevance score
        direct_matches = sum(term in chunk_lower for term in query_terms)
        related_matches = sum(term in chunk_lower for term in search_terms)
        total_score = (direct_matches * 2) + related_matches
        
        if total_score > 0:
            scored_chunks.append((chunk, total_score))
    
    # Log search results
    logger.info(f"Found {len(scored_chunks)} relevant chunks out of {len(chunks)} total chunks")
    if not scored_chunks:
        logger.warning(f"No chunks matched the search terms: {search_terms}")
    
    # Sort by relevance score and return top results
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    results = [chunk for chunk, score in scored_chunks[:max_results]]
    
    # Log sample of results
    if results:
        logger.info(f"Top result preview: {results[0][:200]}...")
        
    return results

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

def process_data(text_content: str) -> List[str]:
    """Process text into smaller, meaningful chunks."""
    if not text_content:
        return []
    
    # Split into sections based on headers
    sections = []
    current_section = []
    lines = text_content.split('\n')
    
    for line in lines:
        if any(line.strip().lower().startswith(header) for header in ['#', 'resource', 'data', 'variable', 'provider']):
            if current_section:
                sections.append('\n'.join(current_section))
            current_section = [line]
        else:
            current_section.append(line)
    
    if current_section:
        sections.append('\n'.join(current_section))
    
    # Process each section into chunks
    chunks = []
    chunk_size = 2000
    overlap = 200
    
    for section in sections:
        words = section.split()
        if len(words) < 50:  # Skip very small sections
            continue
            
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
    
    logger.info(f"Processed {len(chunks)} chunks from text content")
    return chunks

def initialize_data():
    """Initialize data by scraping and processing website content."""
    global CHUNKS
    
    try:
        print("Scraping Terraform Registry websites...")
        all_content = []
        
        # Scrape provider documentation
        for url in PROVIDER_URLS:
            provider_content = scrape_website(url)
            if provider_content:
                all_content.append(provider_content)
                print(f"Successfully scraped provider URL: {url}")
            else:
                print(f"Failed to scrape provider URL: {url}")
        
        # Scrape module documentation
        for url in MODULE_URLS:
            module_content = scrape_website(url)
            if module_content:
                all_content.append(module_content)
                print(f"Successfully scraped module URL: {url}")
            else:
                print(f"Failed to scrape module URL: {url}")
        
        ALL_CONTENT = "\n\n".join(all_content)
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
