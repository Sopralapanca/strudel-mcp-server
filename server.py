import asyncio
import os
from typing import Any
import requests
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from supabase import create_client, Client
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
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
        # Use the feature extraction endpoint
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={
                "inputs": text,
                "options": {"wait_for_model": True}
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # HF returns embeddings in different formats depending on the endpoint
            # For feature-extraction, it's typically a nested list
            if isinstance(result, list):
                # If it's a list of lists (batch), take the first one
                if len(result) > 0 and isinstance(result[0], list):
                    return result[0]
                # If it's already a flat list
                elif len(result) > 0 and isinstance(result[0], (int, float)):
                    return result
            
            # If we get here, unexpected format
            raise ValueError(f"Unexpected response format: {result}")
            
        else:
            error_msg = f"HF API error: {response.status_code} - {response.text}"
            print(error_msg)
            raise Exception(error_msg)
            
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        raise

# Create MCP server
mcp_server = Server("strudel-docs-server")

@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="search_strudel_docs",
            description="Search the Strudel.cc documentation for information about patterns, samples, effects, syntax, and music generation techniques. Use this when you need to know how to use specific Strudel features or understand the API.",
            inputSchema={
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
        )
    ]

@mcp_server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""
    if name == "search_strudel_docs":
        query = arguments.get("query")
        max_results = arguments.get("maxResults", 3)
        
        try:
            # Generate embedding for query using HF API
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
                return [TextContent(
                    type="text",
                    text="No relevant documentation found for your query."
                )]
            
            # Format results
            results = []
            for idx, doc in enumerate(response.data, 1):
                results.append(
                    f"--- Result {idx} (Similarity: {doc['similarity']:.2f}) ---\n"
                    f"{doc['content']}\n"
                )
            
            return [TextContent(
                type="text",
                text="\n".join(results)
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error searching documentation: {str(e)}"
            )]
    
    raise ValueError(f"Unknown tool: {name}")

# Flask app for HTTP/REST API
flask_app = Flask(__name__)
CORS(flask_app)  # Enable CORS for all routes

@flask_app.route('/')
def home():
    return jsonify({
        'status': 'ok',
        'service': 'Strudel MCP Server',
        'version': '1.0.0',
        'endpoints': {
            'health': '/health',
            'tools': '/tools',
            'call_tool': '/call_tool (POST)'
        }
    })

@flask_app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@flask_app.route('/tools', methods=['GET'])
def get_tools():
    """Return the list of available MCP tools in MCP protocol format"""
    try:
        tools_objs = asyncio.run(list_tools())
        tools_list = []
        
        for t in tools_objs:
            tools_list.append({
                'name': t.name,
                'description': t.description,
                'inputSchema': t.inputSchema
            })
        
        return jsonify({
            'tools': tools_list
        }), 200, {
            'Content-Type': 'application/json',
            'X-MCP-Version': '1.0'
        }
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@flask_app.route('/call_tool', methods=['POST'])
def call_tool_http():
    """
    Call a registered MCP tool via HTTP POST.
    
    Expected JSON body: 
    {
        "name": "search_strudel_docs",
        "arguments": {
            "query": "how to use samples",
            "maxResults": 3
        }
    }
    
    Returns JSON:
    {
        "content": [
            {
                "type": "text",
                "text": "..."
            }
        ]
    }
    """
    try:
        # Parse request
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
            
        payload = request.get_json()
        if not payload:
            return jsonify({'error': 'JSON body required'}), 400

        name = payload.get('name')
        arguments = payload.get('arguments', {})

        if not name:
            return jsonify({'error': 'Tool "name" is required'}), 400

        # Call the tool
        outputs = asyncio.run(call_tool(name, arguments))

        # Format response in MCP protocol format
        content_list = []
        for output in outputs:
            content_list.append({
                'type': output.type,
                'text': output.text
            })

        return jsonify({
            'content': content_list
        }), 200, {
            'Content-Type': 'application/json',
            'X-MCP-Version': '1.0'
        }

    except ValueError as ve:
        return jsonify({
            'error': str(ve),
            'type': 'ValueError'
        }), 400
    except Exception as e:
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

def run_flask():
    """Run Flask server"""
    port = int(os.getenv('PORT', 10000))
    flask_app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

async def main():
    """Main function to run both servers"""
    # Start Flask server in background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print(f"HTTP server started on port {os.getenv('PORT', 10000)}")
    
    # Keep the main thread alive
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    asyncio.run(main())