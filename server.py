import asyncio
import os
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from flask import Flask, jsonify, request
import threading
from dotenv import load_dotenv

load_dotenv()


# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL", ""),
    os.getenv("SUPABASE_KEY", "")
)

# Initialize embedding model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

def generate_embedding(text: str) -> list[float]:
    """Generate embedding for a text using local model"""
    embedding = model.encode(text, convert_to_tensor=False)
    return embedding.tolist()

# Create MCP server
app = Server("strudel-docs-server")

@app.list_tools()
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

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls"""
    if name == "search_strudel_docs":
        query = arguments.get("query")
        max_results = arguments.get("maxResults", 3)
        
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

# Flask app for health checks (for Render.com)
flask_app = Flask(__name__)

@flask_app.route('/')
def home():
    return jsonify({
        'status': 'ok',
        'service': 'Strudel MCP Server',
        'version': '1.0.0'
    })

@flask_app.route('/health')
def health():
    return jsonify({'status': 'healthy'})


@flask_app.route('/tools')
def tools():
    """Return the list of registered MCP tools as JSON.

    This calls the module-level async `list_tools` function and converts
    the returned Tool objects into JSON-serializable dictionaries.
    """
    try:
        tools_objs = asyncio.run(list_tools())
        tools_json = []
        for t in tools_objs:
            tools_json.append({
                'name': getattr(t, 'name', None),
                'description': getattr(t, 'description', None),
                'inputSchema': getattr(t, 'inputSchema', None)
            })
        return jsonify({'tools': tools_json})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@flask_app.route('/call_tool', methods=['POST'])
def call_tool_http():
    """Call a registered MCP tool via HTTP POST.

    Expected JSON body: { "name": "tool_name", "arguments": { ... } }
    Returns JSON: { "outputs": [ { "type": "text", "text": "..." }, ... ] }
    """
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({'error': 'JSON body required'}), 400

        name = payload.get('name')
        arguments = payload.get('arguments', {})

        if not name:
            return jsonify({'error': 'Tool "name" is required'}), 400

        # call the async call_tool function and get results
        outputs = asyncio.run(call_tool(name, arguments))

        # Serialize TextContent objects to JSON
        out_json = []
        for o in outputs:
            out_json.append({
                'type': getattr(o, 'type', None),
                'text': getattr(o, 'text', None)
            })

        return jsonify({'outputs': out_json})

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_flask():
    """Run Flask server in a separate thread"""
    port = int(os.getenv('PORT', 3000))
    flask_app.run(host='0.0.0.0', port=port, debug=False)

async def main():
    """Main function to run MCP server"""
    # Start Flask server in background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print(f"HTTP server started on port {os.getenv('PORT', 3000)}")
    
    # Run MCP server on stdio
    async with stdio_server() as (read_stream, write_stream):
        print("Strudel MCP server running on stdio", flush=True)
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())