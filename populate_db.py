import os
import re
from server import generate_embedding
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

load_dotenv()

# Initialize Supabase
url: str = os.environ.get("SUPABASE_URL", "")
key: str = os.environ.get("SUPABASE_KEY", "")
if not url or not key:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables.")

supabase: Client = create_client(url, key)

def chunk_documentation(markdown_text: str) -> list:
    """Split documentation into meaningful chunks"""
    chunks = []
    
    # Split by ## headers (main sections) using multiline regex
    # This mimics markdown.split(/^## /m)
    sections = re.split(r'^## ', markdown_text, flags=re.MULTILINE)
    
    # Iterate starting from index 1 (skipping the pre-header content), matching JS logic
    for section_content in sections[1:]:
        section = '## ' + section_content
        
        # If section is too large (>2000 chars), split further
        if len(section) > 2000:
            # Split by ### subheaders
            subsections = re.split(r'^### ', section, flags=re.MULTILINE)
            
            # Keep main header with intro (index 0)
            chunks.append('## ' + subsections[0])
            
            # Add subsections
            for sub in subsections[1:]:
                chunks.append('### ' + sub)
        else:
            chunks.append(section)
            
    # Filter out short chunks
    return [chunk for chunk in chunks if len(chunk.strip()) > 50]

def populate_vector_db():
    try:
        print('Reading Strudel documentation...')
        # Assuming file is in the same directory
        with open('strudel-docs.md', 'r', encoding='utf-8') as f:
            docs = f.read()
            
        print('Chunking documentation...')
        chunks = chunk_documentation(docs)
        print(f'Created {len(chunks)} chunks')
        
        print('Generating embeddings and storing in Supabase...')
        
        for i, chunk in enumerate(chunks):
            print(f'Processing chunk {i + 1}/{len(chunks)}...')
            
            # Generate embedding locally
            embedding = generate_embedding(chunk)
            
            # Store in Supabase
            data = {
                "content": chunk,
                "embedding": embedding
            }
            
            response = supabase.table('strudel_docs').insert(data).execute()
            
            # supabase-py raises exceptions on error usually, but we can check response
            # Note: older versions returned {data, error}, newer raise postgrest.exceptions.APIError
            
        print('✅ Successfully populated vector database!')
        
        # Test a search
        print('\nTesting search...')
        test_query = "How do I play samples?"
        
        # For BGE query embedding, some versions recommend adding an instruction prefix
        # However, keeping it simple to match previous logic:
        test_embedding = generate_embedding(test_query)
        
        rpc_params = {
            'query_embedding': test_embedding,
            'match_threshold': 0.5,
            'match_count': 3
        }
        
        response = supabase.rpc('match_documents', rpc_params).execute()
        
        if response.data:
            print(f'Found {len(response.data)} relevant documents for: "{test_query}"')
            if len(response.data) > 0:
                print('\nTop result preview:')
                print(response.data[0]['content'][:200] + '...')
        else:
             print(f'No relevant documents found (or empty response).')

    except Exception as e:
        print('Error:', e)

if __name__ == "__main__":
    populate_vector_db()