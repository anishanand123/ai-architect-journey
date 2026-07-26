from sentence_transformers import SentenceTransformer
import chromadb

# STEP 1: Load embedding model (same as Day 1)
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Model loaded\n")

# STEP 2: Define knowledge base
knowledge_base = [
    {
        "id": "taskflow_what",
        "title": "What is TaskFlow?",
        "content": """TaskFlow is a cloud-based project management tool 
designed for small to medium teams. It allows users to create projects, 
assign tasks, set deadlines, and track progress on a visual Kanban board. 
TaskFlow integrates with popular calendar and email tools, and is 
accessible via web browser and mobile app."""
    },
    {
        "id": "taskflow_password",
        "title": "How do I reset my TaskFlow password?",
        "content": """To reset your TaskFlow password, go to the login 
page and click "Forgot Password". Enter the email address associated 
with your account. You will receive a password reset link valid for 24 
hours. After clicking the link, you can set a new password. If you don't 
receive the email within a few minutes, check your spam folder or contact 
support@taskflow-example.com."""
    },
    {
        "id": "taskflow_pricing",
        "title": "What are TaskFlow's pricing plans?",
        "content": """TaskFlow offers three pricing plans: Free (up to 3 
projects and 5 team members), Pro at $8 per user per month (unlimited 
projects, advanced reporting, and integrations), and Business at $15 per 
user per month (includes priority support, advanced permissions, and custom 
branding). All plans include a 14-day free trial of Pro features."""
    }
]

# STEP 3: Create and populate ChromaDB
print("Setting up ChromaDB collection...")
client = chromadb.Client()
collection = client.create_collection(name="taskflow_kb")

for article in knowledge_base:
    text_to_embed = f"{article['title']}. {article['content']}"
    embedding = model.encode(text_to_embed).tolist()
    
    collection.add(
        ids=[article["id"]],
        embeddings=[embedding],
        documents=[article["content"]],
        metadatas=[{"title": article["title"]}]
    )

print(f"✓ {collection.count()} articles in database\n")

# STEP 4: Create semantic search function (NEW)
def semantic_search(query, top_k=1):
    """
    Search knowledge base semantically.
    
    Args:
        query: User's question
        top_k: Number of results to return (1 = best match, 2 = top 2, etc)
    
    Returns:
        List of matching articles with similarity scores
    """
    # Embed the query
    query_embedding = model.encode(query).tolist()
    
    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Format results
    matches = []
    if results['ids'][0]:
        for i, article_id in enumerate(results['ids'][0]):
            match = {
                "id": article_id,
                "title": results['metadatas'][0][i]['title'],
                "content": results['documents'][0][i],
                "similarity": results['distances'][0][i]  # Lower = more similar
            }
            matches.append(match)
    
    return matches

# STEP 5: Test the search function with various queries
print("="*70)
print("TESTING SEMANTIC SEARCH FUNCTION")
print("="*70 + "\n")

test_cases = [
    ("What is TaskFlow?", "Direct match - exact question"),
    ("Tell me about your tool", "Semantic match - different wording"),
    ("How do I reset my password?", "Direct match - password question"),
    ("I forgot my password", "Semantic match - same meaning, different words"),
    ("What's your pricing?", "Semantic match - pricing question"),
    ("Do you have a free option?", "Semantic match - free tier question"),
    ("Does TaskFlow integrate with Jira?", "No match - out of scope"),
    ("What's the weather today?", "No match - out of scope")
]

for query, description in test_cases:
    print(f"Query: '{query}'")
    print(f"Type: {description}")
    
    # Search
    results = semantic_search(query, top_k=1)
    
    if results:
        result = results[0]
        print(f"✓ Found: {result['title']}")
        print(f"  Similarity: {result['similarity']:.4f} (lower = better)")
        print(f"  Content preview: {result['content'][:80]}...\n")
    else:
        print("✗ No relevant article found\n")

# STEP 6: Compare P2 vs P3 retrieval
print("="*70)
print("P2 vs P3 COMPARISON")
print("="*70 + "\n")

comparison_queries = [
    "How much does it cost?",
    "Can I use for free?",
    "Manage team tasks",
]

print("Query: 'How much does it cost?'\n")
print("P2 (Keyword matching):")
print("  - Looks for keyword 'cost' in articles")
print("  - Might miss 'pricing' articles")
print("  - Accuracy: Medium\n")

print("P3 (Semantic search):")
results = semantic_search("How much does it cost?")
if results:
    print(f"  - Understands 'cost' = 'pricing'")
    print(f"  - Found: {results[0]['title']}")
    print(f"  - Similarity: {results[0]['similarity']:.4f}")
    print(f"  - Accuracy: High\n")

print("="*70)
print("WHY P3 IS BETTER")
print("="*70)
print("""
P2 (Keyword):
  Pros: Fast, simple, deterministic
  Cons: Misses synonyms, poor with natural language

P3 (Semantic):
  Pros: Understands meaning, handles synonyms, works with natural language
  Cons: Slightly slower (still <100ms), needs embedding model

Example:
  P2 can't connect "cost" → "pricing" → "free plan"
  P3 understands all are related to pricing discussion
""")

print("="*70)
print("Day 2 complete!")
print("="*70)