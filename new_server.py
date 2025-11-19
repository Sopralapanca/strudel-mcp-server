import os
import json
import uuid
import requests
from supabase import create_client, Client
from flask import Flask, Response, jsonify, request, stream_with_context
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

print("Server starting with SSE MCP protocol...")

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

# Flask app
flask_app = Flask(__name__)
CORS(flask_app, resources={r"/*": {"origins": "*", "allow_headers": "*"}})

# MCP Server capabilities
SERVER_INFO = {
    "protocolVersion": "2024-11-05",
    "capabilities": {
        "tools": {}
    },
    "serverInfo": {
        "name": "strudel-docs-server",
        "version": "1.0.0"
    }
}

TOOLS = [
    {
        "name": "search_strudel_docs",
        "description": "Search the Strudel.cc documentation for information about patterns, samples, effects, syntax, and music generation techniques.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query (e.g., 'how to use samples', 'mini notation syntax')"
                },
                "maxResults": {
                    "type": "number",
                    "description": "Maximum number of results (default: 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    }
]

def send_sse_message(data):
    """Format data as SSE message"""
    return f"data: {json.dumps(data)}\n\n"


def process_mcp_message(message):
    """Process an MCP JSON-RPC message (returns tuple (response_obj, status_code)).

    For notifications that require no response, returns (None, 204).
    """
    try:
        method = message.get('method')
        params = message.get('params', {})
        msg_id = message.get('id')

        print(f"Received MCP message: {method}")

        # Handle initialize
        if method == 'initialize':
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": SERVER_INFO
            }
            return response, 200

        # Handle tools/list
        elif method == 'tools/list':
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": TOOLS
                }
            }
            return response, 200

        # Handle tools/call
        elif method == 'tools/call':
            tool_name = params.get('name')
            arguments = params.get('arguments', {})

            if tool_name == 'search_strudel_docs':
                query = arguments.get('query')
                max_results = arguments.get('maxResults', 3)

                if not query:
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {
                            "code": -32602,
                            "message": "Missing required argument: query"
                        }
                    }, 400

                try:
                    # Generate embedding
                    query_embedding = generate_embedding(query)

                    # Search database
                    db_response = supabase.rpc(
                        'match_documents',
                        {
                            'query_embedding': query_embedding,
                            'match_threshold': 0.6,
                            'match_count': max_results
                        }
                    ).execute()

                    if not db_response.data:
                        result_text = "No relevant documentation found for your query."
                    else:
                        results = []
                        for idx, doc in enumerate(db_response.data, 1):
                            results.append(
                                f"--- Result {idx} (Similarity: {doc['similarity']:.2f}) ---\n"
                                f"{doc['content']}\n"
                            )
                        result_text = "\n".join(results)

                    response = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": result_text
                                }
                            ]
                        }
                    }
                    return response, 200

                except Exception as e:
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {
                            "code": -32603,
                            "message": f"Error searching documentation: {str(e)}"
                        }
                    }, 500
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }, 400

        # Handle notifications/initialized
        elif method == 'notifications/initialized':
            # No response needed for notifications
            return None, 204

        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }, 400

    except Exception as e:
        print(f"Error handling message: {e}")
        return {
            "jsonrpc": "2.0",
            "id": message.get('id'),
            "error": {
                "code": -32603,
                "message": str(e)
            }
        }, 500

@flask_app.route('/')
def home():
    """Root endpoint (also accepts POST for streamable_http fallback)."""
    if request.method == 'GET':
        return jsonify({
            "status": "ok",
            "service": "Strudel MCP Server",
            "protocol": "MCP with SSE",
            "version": "1.0.0"
        }), 200

    # Allow POST at root to support streamable_http initialize fallback
    if request.method == 'POST':
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400

        message = request.get_json()
        result, status = process_mcp_message(message)

        # notifications return no content
        if status == 204:
            return '', 204

        return jsonify(result), status

@flask_app.route('/health')
def health():
    """Health check"""
    return jsonify({'status': 'healthy'}), 200

@flask_app.route('/sse', methods=['GET', 'POST'])
def sse_endpoint():
    """SSE endpoint for MCP protocol. Sends an absolute `/message` URL in the first `endpoint` event."""

    def generate():
        # Build an absolute message endpoint URL so clients know where to POST
        try:
            endpoint_url = request.host_url.rstrip('/') + '/message'
        except Exception:
            endpoint_url = '/message'

        # Send endpoint event first (required by MCP spec)
        yield f"event: endpoint\ndata: {endpoint_url}\n\n"

        # Keep connection alive
        while True:
            yield ": keepalive\n\n"
            import time
            time.sleep(30)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )

@flask_app.route('/message', methods=['POST'])
def message_endpoint():
    """Handle MCP messages"""
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400

    message = request.get_json()
    result, status = process_mcp_message(message)

    if status == 204:
        return '', 204

    return jsonify(result), status

if __name__ == "__main__":
    port = int(os.getenv('PORT', 3000))
    print(f"Starting MCP SSE server on port {port}")
    print("SSE endpoint: /sse")
    print("Message endpoint: /message")
    flask_app.run(host='0.0.0.0', port=port, debug=False, threaded=True)