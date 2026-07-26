from sentence_transformers import SentenceTransformer
import chromadb

# STEP 1: Load embedding model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Model loaded\n")

# STEP 2: Define knowledge base (same as P2)
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

# STEP 3: Create embeddings for each article
print("Creating embeddings for knowledge base articles...\n")
embeddings_list = []

for article in knowledge_base:
    # Combine title and content for embedding
    text_to_embed = f"{article['title']}. {article['content']}"
    
    # Create embedding (text → vector)
    embedding = model.encode(text_to_embed)
    
    embeddings_list.append({
        "id": article["id"],
        "title": article["title"],
        "content": article["content"],
        "embedding": embedding.tolist()  # Convert to list for storage
    })
    
    print(f"✓ Embedded: {article['title']}")
    print(f"  Vector size: {len(embedding)} dimensions")
    print(f"  First 5 values: {embedding[:5]}\n")

# STEP 4: Initialize ChromaDB collection
print("Initializing ChromaDB...")
client = chromadb.Client()
collection = client.create_collection(name="taskflow_kb")
print("✓ ChromaDB collection created\n")

# STEP 5: Add embeddings to ChromaDB
print("Adding embeddings to ChromaDB...\n")
for item in embeddings_list:
    collection.add(
        ids=[item["id"]],
        embeddings=[item["embedding"]],
        documents=[item["content"]],
        metadatas=[{"title": item["title"]}]
    )
    print(f"✓ Stored: {item['title']}")

print(f"\n✓ Total articles in database: {collection.count()}")

# STEP 6: Test semantic search
print("\n" + "="*60)
print("TESTING SEMANTIC SEARCH")
print("="*60 + "\n")

test_queries = [
    "What is TaskFlow?",
    "How do I reset my password?",
    "What does it cost?",
    "Do you have a free plan?"
]

for query in test_queries:
    print(f"Query: '{query}'")
    
    # Embed the query (same way as articles)
    query_embedding = model.encode(query).tolist()
    
    # Search ChromaDB for most similar articles
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1  # Get top 1 result
    )
    
    # Display results
    if results['ids'][0]:
        article_id = results['ids'][0][0]
        title = results['metadatas'][0][0]['title']
        distance = results['distances'][0][0]  # Similarity score (lower = more similar)
        
        print(f"  → Found: {title}")
        print(f"  → Similarity score: {distance:.4f}")
    
    print()

print("="*60)
print("Day 1 complete!")
print("="*60)