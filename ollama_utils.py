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
    """Enhanced code generation with health checks and fallbacks."""
    try:
        logger.info(f"Starting code generation for query: {query}")
        logger.info(f"Using model: {model_name}")
        logger.info(f"Cloud provider: {cloud_provider}")
        
        # Check server health before proceeding
        if not check_ollama_connection():
            raise ModelNotAvailableError("Ollama server is not responding")
            
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
        
        # Limit context size based on query length
        max_context_tokens = min(8192, 16384 - len(query))
        context_chunks = context_chunks[:max_context_tokens // 100]  # Rough estimate of tokens per chunk
        
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

# Update session configuration with longer timeouts
session = requests.Session()
retry_strategy = Retry(
    total=5,  # Increased from 3
    backoff_factor=2,  # Increased from 1
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"],  # Explicitly allow POST
    respect_retry_after_header=True
)
adapter = HTTPAdapter(
    max_retries=retry_strategy,
    pool_connections=20,  # Increased from 10
    pool_maxsize=20,     # Increased from 10
    pool_block=True
)
session.mount("http://", adapter)
session.mount("https://", adapter)

def generate_with_retry(model_name: str, prompt: str, max_retries: int = 5, base_delay: float = 2.0) -> str:
    """Generate code using the more efficient generate endpoint."""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1} of {max_retries}")
            
            # Add explicit formatting instructions to prompt
            formatted_prompt = f"{prompt}\n\nPlease provide the complete Terraform code wrapped in ```hcl blocks."
            
            payload = {
                "model": model_name,
                "prompt": formatted_prompt,
                "stream": False,
                "raw": False,  # Changed to false to get formatted response
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": 8192,
                    "num_predict": 4096,
                    "stop": ["</code>", "```\n"],
                    "repeat_penalty": 1.1
                }
            }
            
            logger.info(f"Sending request to Ollama generate API")
            response = session.post(
                f"{OLLAMA_API_URL}/api/generate",
                json=payload,
                timeout=(30, 180)
            )
            
            response.raise_for_status()
            result = response.json()
            
            if 'response' in result:
                content = result['response']
                logger.info("Received response, cleaning code...")
                cleaned_code = clean_code_response(content)
                if cleaned_code:
                    return cleaned_code
                raise CodeGenerationError("Failed to extract valid code from response")
            else:
                raise CodeGenerationError("Invalid response format from API")
                
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
            last_error = e
            
            # Reduce complexity on retry
            if 'options' in payload:
                payload['options']['temperature'] = min(0.9, payload['options']['temperature'] + 0.1)
                payload['options']['num_ctx'] = max(4096, payload['options']['num_ctx'] - 2048)
            
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
    
    raise CodeGenerationError(f"Failed after {max_retries} attempts. Last error: {str(last_error)}")

def clean_code_response(content: str) -> str:
    """Clean and validate the code response with improved parsing."""
    # First try to extract code blocks
    if "```" in content:
        code_blocks = []
        lines = content.split("\n")
        in_code_block = False
        current_block = []
        
        for line in lines:
            if "```" in line:
                if not in_code_block and ("hcl" in line.lower() or "terraform" in line.lower()):
                    in_code_block = True
                elif in_code_block:
                    if current_block:  # Only add non-empty blocks
                        block_content = "\n".join(current_block).strip()
                        if is_valid_terraform(block_content):
                            code_blocks.append(block_content)
                    current_block = []
                    in_code_block = False
            elif in_code_block:
                current_block.append(line)
        
        # Handle any remaining block
        if current_block:
            block_content = "\n".join(current_block).strip()
            if is_valid_terraform(block_content):
                code_blocks.append(block_content)
        
        if code_blocks:
            return "\n\n".join(code_blocks)
    
    # If no valid code blocks found, try to validate the entire content
    if is_valid_terraform(content):
        return content
    
    raise CodeGenerationError("No valid Terraform code found in response")

def is_valid_terraform(content: str) -> bool:
    """Validate if the content looks like Terraform code."""
    required_patterns = [
        r'provider\s+["\w]+\s*{',
        r'resource\s+["\w]+\s+["\w]+\s*{',
        r'variable\s+["\w]+\s*{',
        r'terraform\s*{',
        r'output\s+["\w]+\s*{'
    ]
    
    content_lower = content.lower()
    
    # Check for common Terraform keywords
    if not any(keyword in content_lower for keyword in ['provider', 'resource', 'variable', 'terraform']):
        return False
    
    # Check for valid HCL syntax patterns
    import re
    for pattern in required_patterns:
        if re.search(pattern, content):
            return True
    
    return False

# Add connection health check
def check_ollama_connection() -> bool:
    """Check if Ollama server is responsive."""
    try:
        response = session.get(
            f"{OLLAMA_API_URL}/api/version",
            timeout=(5, 10)
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Ollama server health check failed: {e}")
        return False

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
    # First try to extract code blocks
    if "```" in content:
        code_blocks = []
        lines = content.split("\n")
        in_code_block = False
        current_block = []
        
        for line in lines:
            if "```" in line:
                if not in_code_block and ("hcl" in line.lower() or "terraform" in line.lower()):
                    in_code_block = True
                elif in_code_block:
                    if current_block:  # Only add non-empty blocks
                        block_content = "\n".join(current_block).strip()
                        if is_valid_terraform(block_content):
                            code_blocks.append(block_content)
                    current_block = []
                    in_code_block = False
            elif in_code_block:
                current_block.append(line)
        
        # Handle any remaining block
        if current_block:
            block_content = "\n".join(current_block).strip()
            if is_valid_terraform(block_content):
                code_blocks.append(block_content)
        
        if code_blocks:
            return "\n\n".join(code_blocks)
    
    # If no valid code blocks found, try to validate the entire content
    if is_valid_terraform(content):
        return content
    
    raise CodeGenerationError("No valid Terraform code found in response")

def is_valid_terraform(content: str) -> bool:
    """Validate if the content looks like Terraform code."""
    required_patterns = [
        r'provider\s+["\w]+\s*{',
        r'resource\s+["\w]+\s+["\w]+\s*{',
        r'variable\s+["\w]+\s*{',
        r'terraform\s*{',
        r'output\s+["\w]+\s*{'
    ]
    
    content_lower = content.lower()
    
    # Check for common Terraform keywords
    if not any(keyword in content_lower for keyword in ['provider', 'resource', 'variable', 'terraform']):
        return False
    
    # Check for valid HCL syntax patterns
    import re
    for pattern in required_patterns:
        if re.search(pattern, content):
            return True
    
    return False
