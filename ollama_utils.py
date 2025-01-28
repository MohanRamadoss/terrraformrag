import requests
import ollama
import logging
import traceback  # Add this import
from typing import List, Optional, Dict, Any
import time
import os
import json
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Update Ollama API configuration to use the remote server
OLLAMA_API_URL = "http://209.137.198.220:11434"
logger.info(f"Using remote Ollama server at: {OLLAMA_API_URL}")

class ModelNotAvailableError(Exception):
    """Raised when a requested model is not available."""
    pass

class CodeGenerationError(Exception):
    """Raised when code generation fails."""
    pass

# Define priority models
PRIORITY_MODELS = ["granite-code:20b", "deepseek-r1:14b", "codestral:latest"]

def get_available_models() -> List[str]:
    """Get list of available models from Ollama API directly."""
    try:
        # Make direct request to the Ollama API
        response = requests.get(f"{OLLAMA_API_URL}/api/tags")
        response.raise_for_status()
        
        data = response.json()
        if not data or 'models' not in data:
            logger.warning(f"Unexpected response format from {OLLAMA_API_URL}/api/tags")
            return PRIORITY_MODELS
        
        # Extract model names
        model_names = [
            model['name'] for model in data['models'] 
            if isinstance(model, dict) and 'name' in model
        ]
        
        # Filter for priority models
        available_models = [
            name for name in model_names 
            if name in PRIORITY_MODELS
        ]
        
        if available_models:
            logger.info(f"Found available models: {available_models}")
            return available_models
        
        logger.warning("No priority models found, using defaults")
        return PRIORITY_MODELS
        
    except Exception as e:
        logger.error(f"Error getting models from {OLLAMA_API_URL}: {e}")
        return PRIORITY_MODELS

def validate_model(model_name: str, available_models: List[str]) -> str:
    """Validate model availability and return appropriate model name."""
    if not model_name or model_name not in available_models:
        fallback = available_models[0]
        logger.warning(f"Model {model_name} not available, falling back to {fallback}")
        return fallback
    return model_name

def generate_code_ollama(query: str, context_chunks: List[str], model_name: str, cloud_provider: str = 'aws') -> str:
    """Enhanced code generation with better error handling."""
    try:
        logger.info(f"Starting code generation for query: {query}")
        logger.info(f"Using model: {model_name}")
        logger.info(f"Cloud provider: {cloud_provider}")
        
        # Validate model availability
        available_models = get_available_models()
        if not available_models:
            raise ModelNotAvailableError("No models available from Ollama server")
        
        model_name = validate_model(model_name, available_models)
        logger.info(f"Validated model: {model_name}")
        
        # Check for database-specific query
        is_database_query = any(word in query.lower() for word in ['database', 'postgresql', 'mysql', 'sql'])
        if is_database_query and cloud_provider.lower() != 'azure':
            logger.info("Database query detected, forcing Azure provider")
            cloud_provider = 'azure'
        
        # Prepare complete prompt with context
        prompt = prepare_model_prompt(query, context_chunks, model_name)
        logger.info("Prompt prepared, sending to Ollama API")
        
        # Generate code with retry logic
        response = generate_with_retry(model_name, prompt)
        
        if not response:
            raise CodeGenerationError("Empty response from model")
        
        logger.info("Successfully generated code response")
        return response
        
    except Exception as e:
        logger.error(f"Code generation error: {str(e)}\nTraceback: {traceback.format_exc()}")
        raise CodeGenerationError(f"Failed to generate code: {str(e)}")

# Add session configuration
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
session.mount("http://", adapter)
session.mount("https://", adapter)

def generate_with_retry(model_name: str, prompt: str, max_retries: int = 3, base_delay: float = 1.0) -> str:
    """Generate code with improved retry logic and logging."""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1} of {max_retries}")
            
            # Prepare request payload
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": 4096,
                    "num_predict": 2048,
                }
            }
            
            logger.info(f"Sending request to Ollama API: {OLLAMA_API_URL}")
            response = session.post(  # Use session instead of requests directly
                f"{OLLAMA_API_URL}/api/chat",
                json=payload,
                timeout=60  # Increase timeout
            )
            
            # Log response status
            logger.info(f"Ollama API response status: {response.status_code}")
            response.raise_for_status()
            
            # Parse response
            try:
                result = response.json()
                logger.info("Successfully parsed JSON response")
                
                if 'message' in result and 'content' in result['message']:
                    content = result['message']['content']
                    logger.info("Found content in response")
                    return clean_code_response(content)
                else:
                    raise CodeGenerationError(f"Invalid response format: {result}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                content = response.text
                if content and any(keyword in content.lower() for keyword in ['provider', 'resource', 'terraform']):
                    return clean_code_response(content)
                raise CodeGenerationError(f"Failed to parse response: {str(e)}")
                
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error on attempt {attempt + 1}: {str(e)}")
            last_error = e
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error on attempt {attempt + 1}: {str(e)}")
            last_error = e
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
            last_error = e
            
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Retrying in {delay} seconds...")
            time.sleep(delay)
            continue
    
    error_msg = f"Failed after {max_retries} attempts. Last error: {str(last_error)}"
    logger.error(error_msg)
    raise CodeGenerationError(error_msg)

def prepare_model_prompt(query: str, context_chunks: List[str], model_name: str) -> str:
    """Prepare model-specific prompts with improved structure."""
    base_guidelines = """Write production-ready Terraform code following these requirements:
- Use clear, descriptive resource names and variables
- Include detailed comments explaining each block's purpose
- Follow HashiCorp naming conventions
- Implement proper resource dependencies
- Use variables for configurable values
- Include required provider versions
- Add proper tags for resource management
- Follow security best practices
- Implement error handling
- Use data sources when appropriate"""

    if model_name == "granite-code:20b":
        system_prompt = f"""You are a senior Terraform infrastructure engineer specializing in cloud infrastructure.
Focus on writing production-grade, maintainable Terraform code.

{base_guidelines}

Required structure:
1. Required providers block with versions
2. Provider configuration
3. Variables with descriptions and types
4. Data sources (if needed)
5. Resources with proper naming and tags
6. Outputs for important values

Example:
```hcl
terraform {{
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }}
  }}
}}

provider "aws" {{
  region = var.aws_region
}}

variable "environment" {{
  description = "Environment name"
  type        = string
}}

resource "aws_instance" "example" {{
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.instance_type
  
  tags = {{
    Name        = "${{var.environment}}-instance"
    Environment = var.environment
    ManagedBy   = "terraform"
  }}
}}
```"""
        
        context_text = "\n".join(f"- {chunk}" for chunk in context_chunks) if context_chunks else ""
        query_section = f"\nContext from Terraform Registry:\n{context_text}\n" if context_text else ""
        query_section += f"\nGenerate complete Terraform code for: {query}"
        
        return f"{system_prompt}\n{query_section}"

    elif model_name.startswith("deepseek"):
        system_prompt = f"""You are an experienced Terraform developer creating infrastructure as code.

{base_guidelines}

Include in your response:
- Complete provider configuration
- Well-structured resource blocks
- Comprehensive variable definitions
- Clear comments and documentation
- Proper error handling
- Security best practices"""

        context_text = "\n".join(f"- {chunk}" for chunk in context_chunks) if context_chunks else ""
        instruction = f"""<s>[INST] {system_prompt}
{f'Context from Terraform Registry:\n{context_text}\n' if context_text else ''}
Create Terraform code for: {query} [/INST]</s>"""
        
        return instruction

    elif model_name == "codestral:latest":
        system_prompt = f"""You are a Terraform infrastructure expert. 
Create production-ready infrastructure code.

{base_guidelines}

Follow this structure:
1. Provider configuration
2. Required variables
3. Resource definitions
4. Outputs (if needed)
"""
        context_text = "\n".join(f"- {chunk}" for chunk in context_chunks) if context_chunks else ""
        return f"{system_prompt}\n\n{f'Context:\n{context_text}\n' if context_text else ''}Query: {query}"

    else:  # Generic prompt for other models
        system_prompt = f"""As a Terraform expert, write production-ready infrastructure code.

{base_guidelines}"""
        
        context_text = "\n".join(f"- {chunk}" for chunk in context_chunks) if context_chunks else ""
        return f"{system_prompt}\n\n{f'Context:\n{context_text}\n' if context_text else ''}Query: {query}"

    # Add Azure-specific example for database queries
    if "postgresql" in query.lower() or "database" in query.lower():
        azure_db_example = '''
```hcl
resource "azurerm_postgresql_server" "example" {
  name                = "postgresql-server-1"
  location            = azurerm_resource_group.example.location
  resource_group_name = azurerm_resource_group.example.name

  administrator_login          = "psqladminun"
  administrator_login_password = "H@Sh1CoR3!"

  sku_name   = "GP_Gen5_2"
  version    = "11"
  storage_mb = 5120

  backup_retention_days        = 7
  geo_redundant_backup_enabled = true
  auto_grow_enabled           = true

  high_availability {
    mode = "ZoneRedundant"
  }

  public_network_access_enabled    = false
  ssl_enforcement_enabled         = true
  ssl_minimal_tls_version_enforced = "TLS1_2"

  tags = {
    Environment = "Production"
    Managed_By  = "terraform"
  }
}
```'''
        system_prompt += f"\n\nHere's an example of an Azure PostgreSQL configuration:\n{azure_db_example}"
    
    return f"{system_prompt}\n\n{f'Context:\n{context_text}\n' if context_text else ''}Query: {query}"

def clean_code_response(content: str) -> str:
    """Clean and validate the code response with improved parsing."""
    if not any(keyword in content.lower() for keyword in ['provider', 'resource', 'data', 'variable', 'output']):
        raise CodeGenerationError("Generated content does not contain valid Terraform code")

    # Extract code blocks
    if "```" in content:
        code_blocks = []
        lines = content.split("\n")
        in_code_block = False
        current_block = []
        
        for line in lines:
            if "```" in line:
                if "terraform" in line.lower() or "hcl" in line.lower():
                    # Start of Terraform code block
                    in_code_block = True
                    continue
                elif in_code_block:
                    # End of code block
                    if current_block:
                        code_blocks.append("\n".join(current_block))
                    current_block = []
                    in_code_block = False
            elif in_code_block:
                current_block.append(line)
        
        # Handle any remaining block
        if current_block:
            code_blocks.append("\n".join(current_block))
        
        if code_blocks:
            return "\n\n".join(code_blocks)
    
    # If no code blocks found, return the original content if it looks like Terraform code
    if any(keyword in content.lower() for keyword in ['provider', 'resource', 'data', 'variable', 'output']):
        return content
        
    raise CodeGenerationError("Could not extract valid Terraform code from response")