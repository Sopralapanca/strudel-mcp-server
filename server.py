import asyncio
import os
from typing import Any, Dict, List
import requests
from supabase import create_client, Client
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL", ""),
    os.getenv("SUPABASE_KEY", "")
)

# Hugging Face API configuration
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction"

print("Server starting with Hugging Face API for embeddings...")

def generate_embedding(text: str) -> list[float]:
    """Generate embedding using Hugging Face Inference API"""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": text}
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if isinstance(result, list):
                if len(result) > 0 and isinstance(result[0], list):
                    embedding = result[0]
                else:
                    embedding = result
                
                if len(embedding) != 384:
                    raise ValueError(f"Expected 384-dimensional embedding, got {len(embedding)}")
                
                return embedding
            else:
                raise ValueError(f"Unexpected response format: {type(result)}")
        else:
            error_msg = f"HF API error: {response.status_code} - {response.text}"
            print(error_msg)
            raise Exception(error_msg)
            
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        raise

# Flask app for Groq MCP compatibility
flask_app = Flask(__name__)
CORS(flask_app, resources={r"/*": {"origins": "*"}})

# Tool definition
TOOLS_DEFINITION = [
    {
        "name": "search_strudel_docs",
        "description": "Search the Strudel.cc documentation for information about patterns, samples, effects, syntax, and music generation techniques. Use this when you need to know how to use specific Strudel features or understand the API.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing what you want to know about Strudel (e.g., 'how to use samples', 'mini notation syntax', 'effects and modulation')"
                },
                "maxResults": {
                    "type": "number",
                    "description": "Maximum number of relevant documentation chunks to return (default: 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    }
]

@flask_app.route('/')
def home():
    """Root endpoint - MCP server info"""
    return jsonify({
        "protocol": "mcp",
        "version": "1.0",
        "name": "Strudel Documentation MCP Server",
        "description": "Search Strudel.cc documentation using semantic search",
        "capabilities": {
            "tools": True
        }
    }), 200

@flask_app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

@flask_app.route('/mcp/v1/tools', methods=['GET'])
def list_tools():
    """List available tools - OpenAI/Groq MCP format"""
    return jsonify({
        "tools": TOOLS_DEFINITION
    }), 200

@flask_app.route('/mcp/v1/call_tool', methods=['POST'])
def call_tool():
    """
    Call a tool - OpenAI/Groq MCP format
    
    Expected JSON body:
    {
        "name": "search_strudel_docs",
        "arguments": {
            "query": "how to use samples",
            "maxResults": 3
        }
    }
    """
    try:
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json'
            }), 400
            
        payload = request.get_json()
        if not payload:
            return jsonify({'error': 'JSON body required'}), 400

        tool_name = payload.get('name')
        arguments = payload.get('arguments', {})

        if not tool_name:
            return jsonify({'error': 'Tool "name" is required'}), 400

        if tool_name == "search_strudel_docs":
            query = arguments.get("query")
            max_results = arguments.get("maxResults", 3)
            
            if not query:
                return jsonify({
                    'error': 'Missing required argument: query'
                }), 400
            
            try:
                # Generate embedding for query
                query_embedding = generate_embedding(query)
                
                # Search vector database
                response = supabase.rpc(
                    'match_documents',
                    {
                        'query_embedding': query_embedding,
                        'match_threshold': 0.6,
                        'match_count': max_results
                    }
                ).execute()
                
                if not response.data:
                    result_text = "No relevant documentation found for your query."
                else:
                    # Format results
                    results = []
                    for idx, doc in enumerate(response.data, 1):
                        results.append(
                            f"--- Result {idx} (Similarity: {doc['similarity']:.2f}) ---\n"
                            f"{doc['content']}\n"
                        )
                    result_text = "\n".join(results)
                
                return jsonify({
                    "result": result_text
                }), 200
                
            except Exception as e:
                return jsonify({
                    'error': f"Error searching documentation: {str(e)}"
                }), 500
        else:
            return jsonify({
                'error': f'Unknown tool: {tool_name}'
            }), 400

    except Exception as e:
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

# Legacy endpoints for backward compatibility
@flask_app.route('/tools', methods=['GET'])
def tools_legacy():
    """Legacy tools endpoint"""
    return list_tools()

@flask_app.route('/call_tool', methods=['POST'])
def call_tool_legacy():
    """Legacy call_tool endpoint"""
    return call_tool()

def run_flask():
    """Run Flask server"""
    port = int(os.getenv('PORT', 10000))
    print(f"Starting MCP server on port {port}")
    print(f"MCP endpoints available at:")
    print(f"  - GET  /mcp/v1/tools")
    print(f"  - POST /mcp/v1/call_tool")
    flask_app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

if __name__ == "__main__":
    run_flask()