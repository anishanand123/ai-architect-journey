import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai

# STEP 1: Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

print("Loading models...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Models loaded\n")

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

# STEP 3: Initialize ChromaDB
print("Initializing ChromaDB...")
client = chromadb.Client()
collection = client.create_collection(name="taskflow_kb")

for article in knowledge_base:
    text_to_embed = f"{article['title']}. {article['content']}"
    embedding = embedding_model.encode(text_to_embed).tolist()
    
    collection.add(
        ids=[article["id"]],
        embeddings=[embedding],
        documents=[article["content"]],
        metadatas=[{"title": article["title"]}]
    )

print(f"✓ {collection.count()} articles indexed\n")

# STEP 4: Create semantic search function (from Day 2)
def semantic_search(query, top_k=1, similarity_threshold=0.5):
    """
    Search knowledge base semantically.
    
    Args:
        query: User's question
        top_k: Number of results to return
        similarity_threshold: Only return results with similarity < threshold
    
    Returns:
        List of matching articles or empty list if no good matches
    """
    query_embedding = embedding_model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    matches = []
    if results['ids'][0]:
        for i, article_id in enumerate(results['ids'][0]):
            similarity = results['distances'][0][i]
            
            # Only include if similarity is good (lower score = better match)
            if similarity < similarity_threshold:
                match = {
                    "id": article_id,
                    "title": results['metadatas'][0][i]['title'],
                    "content": results['documents'][0][i],
                    "similarity": similarity
                }
                matches.append(match)
    
    return matches

# STEP 5: Conversation history
conversation_history = []

# STEP 6: Main chat function (combines search + generation)
def chat(user_message):
    """
    Full RAG pipeline:
    1. Retrieve relevant articles
    2. Build context
    3. Generate with Gemini
    4. Add citations
    5. Store in history
    """
    
    # Step 1: Retrieve
    retrieved_articles = semantic_search(user_message, top_k=1, similarity_threshold=1.5)
    
    # Step 2: Build system prompt
    system_prompt = """You are a helpful TaskFlow customer support assistant.
Answer questions using the knowledge base articles provided.
Be friendly and professional. If you don't know the answer, say so."""
    
    # Step 3: Build context
    context = system_prompt
    
    if retrieved_articles:
        article = retrieved_articles[0]
        context += f"\n\nKNOWLEDGE BASE ARTICLE:\n{article['title']}\n{article['content']}"
    else:
        context += "\n\nNo relevant knowledge base articles found."
    
    # Add conversation history
    if conversation_history:
        context += "\n\nCONVERSATION HISTORY:"
        for exchange in conversation_history:
            context += f"\nUser: {exchange['user']}\nAssistant: {exchange['assistant']}"
    
    # Add current question
    context += f"\n\nUser: {user_message}\nAssistant:"
    
    # Step 4: Send to Gemini
    response = model.generate_content(context)
    assistant_message = response.text
    
    # Step 5: Add citations
    if retrieved_articles:
        article = retrieved_articles[0]
        assistant_message += f"\n\n[Source: {article['title']}]"
    
    # Step 6: Store in history
    conversation_history.append({
        "user": user_message,
        "assistant": assistant_message
    })
    
    return assistant_message

# STEP 7: Interactive chat loop
print("="*70)
print("TaskFlow Support Chatbot (P3 - Semantic RAG)")
print("="*70)
print("Type 'quit' to exit\n")

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    
    if not user_input:
        continue
    
    response = chat(user_input)
    print(f"\nBot: {response}\n")