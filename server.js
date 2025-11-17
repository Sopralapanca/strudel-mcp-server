import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { createClient } from '@supabase/supabase-js';
import { pipeline } from '@xenova/transformers';
import dotenv from 'dotenv';
import express from 'express';

dotenv.config();

// Initialize clients
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_KEY
);

// Initialize local embedding model
console.log('Loading embedding model...');
const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
console.log('Model loaded!');

async function generateEmbedding(text) {
  const output = await embedder(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

// Create MCP Server
const server = new Server(
  {
    name: 'strudel-docs-server',
    version: '1.0.0',
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Define the search tool
server.setRequestHandler('tools/list', async () => {
  return {
    tools: [
      {
        name: 'search_strudel_docs',
        description: 'Search the Strudel.cc documentation for information about patterns, samples, effects, syntax, and music generation techniques. Use this when you need to know how to use specific Strudel features or understand the API.',
        inputSchema: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: 'The search query describing what you want to know about Strudel (e.g., "how to use samples", "mini notation syntax", "effects and modulation")',
            },
            maxResults: {
              type: 'number',
              description: 'Maximum number of relevant documentation chunks to return (default: 3)',
              default: 3,
            },
          },
          required: ['query'],
        },
      },
    ],
  };
});

// Handle tool calls
server.setRequestHandler('tools/call', async (request) => {
  if (request.params.name === 'search_strudel_docs') {
    const { query, maxResults = 3 } = request.params.arguments;

    try {
      // Generate embedding for the query
      const embedding = await generateEmbedding(query);

      
      // Search vector database
      const { data, error } = await supabase.rpc('match_documents', {
        query_embedding: embedding,
        match_threshold: 0.6,
        match_count: maxResults,
      });

      if (error) {
        throw error;
      }

      // Format results
      const results = data
        .map((doc, index) => {
          return `--- Result ${index + 1} (Similarity: ${doc.similarity.toFixed(2)}) ---\n${doc.content}\n`;
        })
        .join('\n');

      return {
        content: [
          {
            type: 'text',
            text: results || 'No relevant documentation found for your query.',
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error searching documentation: ${error.message}`,
          },
        ],
        isError: true,
      };
    }
  }

  throw new Error(`Unknown tool: ${request.params.name}`);
});

// HTTP server for Render.com health checks
const app = express();
const PORT = process.env.PORT || 3000;

app.get('/', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'Strudel MCP Server',
    version: '1.0.0'
  });
});

app.get('/health', (req, res) => {
  res.json({ status: 'healthy' });
});

// Start HTTP server
app.listen(PORT, () => {
  console.log(`HTTP server listening on port ${PORT}`);
});

// Start MCP server on stdio
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('Strudel MCP server running on stdio');
}

main().catch((error) => {
  console.error('Server error:', error);
  process.exit(1);
});