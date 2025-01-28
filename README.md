# Terraform Code Generator with RAG System

A web-based application that generates Terraform code using Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG).

## Requirements

### Python Packages
```bash
pip install -r requirements.txt
```

Required packages:
- Flask==2.0.1
- Flask-Limiter==3.1.0
- Flask-Caching==2.0.0
- requests==2.31.0
- beautifulsoup4==4.12.2
- chromadb==0.4.18
- sentence-transformers==2.2.2
- ollama==0.1.6
- python-dotenv==1.0.0

### External Services
- Ollama server running at http://xxxxxx:11434
- Supported LLM models:
  - granite-code:20b (preferred)
  - deepseek-r1:14b (alternative)

## System Architecture

```
                                    ┌─────────────────┐
                                    │  Terraform      │
                                    │  Registry Docs  │
                                    └────────┬────────┘
                                            │
                                            ▼
┌─────────────┐    ┌──────────────┐    ┌────────┐    ┌──────────────┐
│   Web UI    │◄──►│  Flask App   │◄──►│  RAG   │◄──►│ ChromaDB     │
│  (Browser)  │    │  (Backend)   │    │ Engine │    │ (Embeddings) │
└─────────────┘    └──────┬───────┘    └────┬───┘    └──────────────┘
                          │                  │
                          ▼                  ▼
                   ┌──────────────┐    ┌──────────────┐
                   │    Query     │    │    Ollama    │
                   │   History    │    │    Server    │
                   └──────────────┘    └──────────────┘
```

## Functional Flow

1. **Data Initialization**
   - Scrapes Terraform Registry documentation
   - Processes text into chunks
   - Stores embeddings in ChromaDB
   - Initializes available LLM models

2. **User Query Processing**
   - User submits query through web interface
   - System performs either:
     - Keyword-based search
     - Semantic search using ChromaDB

3. **Code Generation**
   - Relevant context is retrieved
   - Context + query sent to LLM
   - Code is generated with retry logic
   - Response is cleaned and formatted

4. **History Management**
   - Successful queries are saved
   - Includes timestamp, query, response
   - Accessible through web interface

## Key Components

### 1. RAG Utils (`rag_utils.py`)
- Handles document scraping
- Manages text chunking
- Interfaces with ChromaDB
- Provides search functionality

### 2. Ollama Utils (`ollama_utils.py`)
- Manages LLM interactions
- Handles model selection
- Implements retry logic
- Formats prompts

### 3. Flask App (`app.py`)
- Web server implementation
- Route handling
- Rate limiting
- Caching
- Error handling

### 4. Web Interface (`templates/index.html`)
- User interface
- Code display
- Model selection
- Search type toggle

## Usage

1. **Start the Application**
```bash
python app.py
```

2. **Access the Interface**
- Open browser to `http://localhost:5000`

3. **Generate Code**
- Enter Terraform requirements
- Select model and search type
- Click "Generate Code"

4. **View Results**
- Generated code appears in code box
- Relevant context shown below
- Recent queries visible in sidebar

## Error Handling

- Rate limiting: 100 requests per day, 10 per minute
- Model fallback if preferred model unavailable
- Retry logic for API calls
- Comprehensive error logging

## Data Flow

```
Query Input → Text Search → Context Retrieval → LLM Prompt → 
Code Generation → Response Cleaning → Display
```

## Best Practices

1. **Queries**
   - Be specific about requirements
   - Include cloud provider
   - Specify resource types
   - Mention any special configurations

2. **Model Selection**
   - granite-code:20b: Best for complex infrastructure
   - deepseek-r1:14b: Good for general Terraform code

3. **Search Types**
   - Keyword: Fast, good for exact matches
   - Semantic: Better for conceptual queries

## Maintenance

- Logs stored in `app.log`
- Query history in `query_history/`
- ChromaDB data in `chroma_db/`

## Security Notes

- Rate limiting prevents abuse
- Input validation implemented
- Error messages sanitized
- API endpoints protected


Here are some example questions/prompts you can type into the Terraform Registry RAG system to generate useful Terraform code:

1. Infrastructure-related questions:
```
- "Create an AWS EC2 instance with a t2.micro type"
- "How to setup an Azure Storage Account with blob container"
- "Generate code for AWS S3 bucket with versioning enabled"
- "Create a Google Cloud VM instance with debian image"
```

2. Network-related questions:
```
- "Setup AWS VPC with public and private subnets"
- "Create Azure Virtual Network with network security group"
- "Configure AWS security group for web application"
- "Setup GCP firewall rules for allowing HTTP traffic"
```

3. Database-related questions:
```
- "Create an AWS RDS instance with MySQL"
- "Setup Azure PostgreSQL database with high availability"
- "Deploy MongoDB Atlas cluster using Terraform"
- "Create AWS DynamoDB table with auto-scaling"
```

4. Monitoring and logging:
```
- "Setup AWS CloudWatch alarms for EC2 instances"
- "Configure Azure Application Insights"
- "Create AWS CloudWatch Log groups"
- "Setup Google Cloud monitoring for Kubernetes cluster"
```

5. Security and compliance:
```
- "Create IAM roles and policies for AWS Lambda"
- "Setup Azure Key Vault with access policies"
- "Configure AWS KMS encryption for S3 bucket"
- "Create GCP service accounts with specific permissions"
```

6. Container and orchestration:
```
- "Deploy AWS EKS cluster with node groups"
- "Setup Azure AKS cluster with autoscaling"
- "Create AWS ECS cluster with Fargate"
- "Configure GKE cluster with custom node pools"
```

Best practices for asking questions:
1. Be specific about the cloud provider (AWS, Azure, GCP)
2. Mention any specific configurations or requirements
3. Include version requirements if needed
4. Specify any tags or naming conventions
5. Mention any security or compliance requirements

Example of a detailed query:
```
"Create an AWS EC2 instance with t2.micro type, running Amazon Linux 2, with security group allowing SSH access from my IP only, and tag it with environment=development"
```

The system will use the context from the Terraform Registry to generate appropriate code based on your query.