from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from functools import wraps  # Add this import
import logging
from datetime import datetime
import json
from pathlib import Path
from rag_utils import (
    retrieve_relevant_chunks, 
    generate_code_ollama, 
    CHUNKS,
    semantic_search,
    get_available_models  # Add this import
)
from ollama_utils import ModelNotAvailableError, CodeGenerationError
import subprocess
import tempfile
import os
from models import QueryRequest
import traceback
import requests
from redis import Redis

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)  # Add this line to create logger instance

app = Flask(__name__, static_folder='static')

# Add check for CHUNKS initialization
if not CHUNKS:
    print("Warning: No content chunks available. Check if scraping was successful.")

# Configure caching with increased limits
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 7200,  # Increased cache timeout
    'CACHE_THRESHOLD': 1000  # Increased cache entries
})

# Configure response size limits
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Set to 16MB

# Configure Redis for rate limiting
redis_client = Redis(
    host='localhost',  # Change this if Redis is on a different host
    port=6379,
    db=0,
    socket_timeout=5
)

# Update limiter configuration with Redis storage
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day", "10 per minute"],
    storage_uri="redis://localhost:6379",  # Use Redis for storage
    storage_options={"socket_timeout": 5}
)

# Create history directory
HISTORY_DIR = Path("query_history")
HISTORY_DIR.mkdir(exist_ok=True)

# Get available models and ensure priority models are first
AVAILABLE_MODELS = get_available_models()
if not any(model in AVAILABLE_MODELS for model in ["granite-code:20b", "deepseek-r1:14b", "codestral:latest"]):
    logging.warning("None of the preferred models (granite-code:20b, deepseek-r1:14b, codestral:latest) are available")

# Update app configuration
app.config.update(
    PROPAGATE_EXCEPTIONS = True,
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024,  # Increased to 32MB
    REQUEST_TIMEOUT = 300  # 5 minutes timeout
)

# Add timeout handling decorator
def timeout_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.Timeout:
            logger.error("Request timed out")
            return jsonify({
                'error': 'The request timed out. Please try again with a simpler query or smaller context.'
            }), 504
    return wrapper

# Update the generate route to handle pagination
@app.route('/api/generate', methods=['POST'])
@limiter.limit("10 per minute")
@timeout_handler
def api_generate():
    try:
        data = QueryRequest(**request.get_json())
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 5000, type=int)  # Increased chunk size
        
        # Get relevant context with pagination
        if data.search_type == 'semantic':
            relevant_context = semantic_search(data.query, n_results=10)  # Increased results
        else:
            relevant_context = retrieve_relevant_chunks(data.query, CHUNKS)
            
        # Generate code with increased context
        response_code = generate_code_ollama(
            query=data.query,
            context_chunks=relevant_context,
            model_name=data.model_name,
            cloud_provider=cloud_provider
        )
        
        # Split response for pagination if needed
        total_length = len(response_code)
        start = (page - 1) * per_page
        end = start + per_page
        paginated_code = response_code[start:end]
        
        return jsonify({
            'code': paginated_code,
            'context': relevant_context,
            'pagination': {
                'total_length': total_length,
                'current_page': page,
                'per_page': per_page,
                'total_pages': (total_length + per_page - 1) // per_page
            }
        })
        
    except ModelNotAvailableError as e:
        logger.error(f"Model error: {str(e)}")
        return jsonify({'error': f"Model error: {str(e)}"}), 500
    except CodeGenerationError as e:
        logger.error(f"Generation error: {str(e)}")
        return jsonify({'error': f"Code generation failed: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': "An unexpected error occurred"}), 500

@app.route('/', methods=['GET', 'POST'])
@limiter.limit("100 per day")
def index():
    try:
        query = ""
        response_code = ""
        relevant_context = []
        cloud_provider = "aws"  # Add default cloud provider
        model_name = next((m for m in ["granite-code:20b", "deepseek-r1:14b", "codestral:latest"] 
                          if m in AVAILABLE_MODELS), AVAILABLE_MODELS[0])
        search_type = "keyword"

        if request.method == 'POST':
            try:
                logger.info(f"Received POST request with data: {request.form}")
                
                query = request.form.get('query', '')
                cloud_provider = request.form.get('cloud_provider', 'azure')
                model_name = request.form.get('model_select', model_name)  # Update model_name from form
                
                if not query:
                    raise ValueError("Query cannot be empty")
                
                # Create QueryRequest object
                query_data = QueryRequest(
                    query=query,
                    model_name=request.form.get('model_select', model_name),
                    cloud_provider=cloud_provider,
                    search_type=request.form.get('search_type', 'keyword')
                )
                
                logger.info(f"Processing query with model {query_data.model_name} for {cloud_provider}")
                
                if not CHUNKS:
                    logger.error("No content chunks available")
                    raise Exception("Content data not initialized properly")
                
                # Get relevant context
                try:
                    relevant_context = semantic_search(query) if query_data.search_type == 'semantic' \
                        else retrieve_relevant_chunks(query, CHUNKS)
                    logger.info(f"Found {len(relevant_context)} relevant context chunks")
                    
                    if not relevant_context:
                        logger.warning("No relevant context found, proceeding with empty context")
                except Exception as e:
                    logger.error(f"Error retrieving context: {str(e)}")
                    relevant_context = []
                
                # Generate code
                try:
                    response_code = generate_code_ollama(
                        query=query_data.query,
                        context_chunks=relevant_context,
                        model_name=query_data.model_name,
                        cloud_provider=query_data.cloud_provider
                    )
                    
                    if not response_code:
                        raise CodeGenerationError("No code was generated")
                    
                    logger.info("Code generation successful")
                    save_history(query, response_code, query_data.model_name)
                    
                except Exception as e:
                    logger.error(f"Code generation error: {str(e)}\nTraceback: {traceback.format_exc()}")
                    raise CodeGenerationError(f"Failed to generate code: {str(e)}")
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error processing request: {error_msg}\nTraceback: {traceback.format_exc()}")
                response_code = f"Error: {error_msg}"
                
        return render_template('index.html',
                             query=query,
                             response_code=response_code,
                             relevant_context=relevant_context,
                             model_name=model_name,
                             cloud_provider=cloud_provider,  # Add cloud_provider to template context
                             search_type=search_type,
                             history=get_recent_history(),
                             AVAILABLE_MODELS=AVAILABLE_MODELS)
                             
    except Exception as e:
        logger.error(f"Unhandled error: {traceback.format_exc()}")
        return render_template('error.html', error=str(e)), 500

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/api/copy-code', methods=['POST'])
@limiter.limit("60 per minute")
def copy_code():
    try:
        data = request.get_json()
        code = data.get('code', '')
        # Add to history or perform other operations
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/features')
def features():
    return render_template('features.html')

def save_history(query, response, model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = HISTORY_DIR / f"{timestamp}.json"
    
    history_data = {
        "timestamp": timestamp,
        "query": query,
        "response": response,
        "model": model_name
    }
    
    with open(history_file, 'w') as f:
        json.dump(history_data, f)

def get_recent_history(limit=5):
    history_files = sorted(HISTORY_DIR.glob("*.json"), reverse=True)[:limit]
    history = []
    
    for file in history_files:
        with open(file) as f:
            history.append(json.load(f))
    
    return history

@app.errorhandler(429)
def ratelimit_handler(e):
    return render_template('error.html', 
                         error="Rate limit exceeded. Please try again later."), 429

def validate_terraform_code(code):
    """Validate Terraform code using terraform fmt and validate commands."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tf_file = os.path.join(tmpdir, "main.tf")
        with open(tf_file, "w") as f:
            f.write(code)
        
        try:
            # Format check
            subprocess.run(["terraform", "fmt", "-check", tf_file], check=True)
            # Validation
            subprocess.run(["terraform", "init"], cwd=tmpdir, check=True)
            subprocess.run(["terraform", "validate"], cwd=tmpdir, check=True)
            return True, "Code validation successful"
        except subprocess.CalledProcessError as e:
            return False, f"Validation error: {str(e)}"

@app.route('/api/validate', methods=['POST'])
def validate_code():
    code = request.json.get('code')
    is_valid, message = validate_terraform_code(code)
    return jsonify({
        'valid': is_valid,
        'message': message
    })

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    feedback_data = request.json
    feedback_data['timestamp'] = datetime.now().isoformat()
    
    # Store feedback in a JSON file
    with open(HISTORY_DIR / 'feedback.json', 'a') as f:
        json.dump(feedback_data, f)
        f.write('\n')
    
    return jsonify({'status': 'success'})

@app.route('/api/export', methods=['POST'])
def export_code():
    code = request.json.get('code')
    format = request.json.get('format', 'tf')
    
    if format == 'json':
        # Convert HCL to JSON format
        with tempfile.TemporaryDirectory() as tmpdir:
            tf_file = os.path.join(tmpdir, "main.tf")
            with open(tf_file, "w") as f:
                f.write(code)
            try:
                result = subprocess.run(
                    ["terraform", "show", "-json"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True
                )
                return Response(
                    result.stdout,
                    mimetype='application/json',
                    headers={'Content-Disposition': 'attachment; filename=terraform-code.tf.json'}
                )
            except subprocess.CalledProcessError:
                return jsonify({'error': 'Failed to convert to JSON'}), 500
    
    # Default .tf format
    return Response(
        code,
        mimetype='text/plain',
        headers={'Content-Disposition': 'attachment; filename=terraform-code.tf'}
    )

# Modify the existing generate_code_ollama function to handle multiple providers
def generate_code_ollama(query, context_chunks, model_name, cloud_provider='aws'):
    """Local wrapper for the Ollama code generation."""
    try:
        # Import here to avoid circular imports
        from ollama_utils import generate_code_ollama as ollama_generate
        
        logger.info(f"Calling Ollama code generation with model {model_name}")
        try:
            code = ollama_generate(query, context_chunks, model_name, cloud_provider)
            if not code:
                raise CodeGenerationError("Empty response from model")
            return code
        except Exception as e:
            logger.error(f"Error in Ollama generate: {str(e)}")
            raise CodeGenerationError(f"Code generation failed: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error in generate_code_ollama wrapper: {str(e)}")
        raise CodeGenerationError(f"Code generation failed: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
