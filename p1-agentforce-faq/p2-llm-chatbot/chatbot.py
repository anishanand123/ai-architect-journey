import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

# KNOWLEDGE BASE
knowledge_base = {
    "what_is_taskflow": {
        "title": "What is TaskFlow?",
        "content": """TaskFlow is a cloud-based project management tool 
designed for small to medium teams. It allows users to create projects, 
assign tasks, set deadlines, and track progress on a visual Kanban board. 
TaskFlow integrates with popular calendar and email tools, and is 
accessible via web browser and mobile app.""",
        "keywords": ["taskflow", "project management", "features", "what is", "tool"]
    },
    "password_reset": {
        "title": "How do I reset my TaskFlow password?",
        "content": """To reset your TaskFlow password, go to the login 
page and click "Forgot Password". Enter the email address associated 
with your account. You will receive a password reset link valid for 24 
hours. After clicking the link, you can set a new password. If you don't 
receive the email within a few minutes, check your spam folder or contact 
support@taskflow-example.com.""",
        "keywords": ["password", "reset", "login", "account", "forgot"]
    },
    "pricing": {
        "title": "What are TaskFlow's pricing plans?",
        "content": """TaskFlow offers three pricing plans: Free (up to 3 
projects and 5 team members), Pro at $8 per user per month (unlimited 
projects, advanced reporting, and integrations), and Business at $15 per 
user per month (includes priority support, advanced permissions, and custom 
branding). All plans include a 14-day free trial of Pro features.""",
        "keywords": ["pricing", "plans", "cost", "free", "pro", "business"]
    }
}

def find_relevant_article(user_query):
    """Find most relevant KB article by keyword matching"""
    user_query_lower = user_query.lower()
    best_match = None
    best_score = 0
    
    for article_key, article in knowledge_base.items():
        score = 0
        for keyword in article["keywords"]:
            if keyword in user_query_lower:
                score += 1
        if score > best_score:
            best_score = score
            best_match = article
    
    return best_match if best_score > 0 else None

# CONVERSATION HISTORY — stores all previous messages
conversation_history = []

def chat(user_message):
    """
    Process a user message and generate a response.
    
    Steps:
    1. Find relevant KB article
    2. Build context (system prompt + history + relevant article)
    3. Send to Gemini
    4. Store exchange in history
    5. Return response
    """
    
    # Step 1: Retrieve
    relevant_article = find_relevant_article(user_message)
    
    # Step 2: Build context
    system_prompt = """You are a helpful TaskFlow customer support assistant.
Answer questions using the provided knowledge base article.
Be friendly and professional. If you don't know the answer, say so."""
    
    context = system_prompt
    
    # Add the relevant article if found
    if relevant_article:
        context += f"\n\nRELEVANT ARTICLE:\n{relevant_article['title']}\n{relevant_article['content']}"
    else:
        context += "\n\nNo relevant article found in knowledge base."
    
    # Add conversation history for context
    if conversation_history:
        context += "\n\nCONVERSATION HISTORY:"
        for exchange in conversation_history:
            context += f"\nUser: {exchange['user']}\nAssistant: {exchange['assistant']}"
    
    # Add current user message
    context += f"\n\nUser: {user_message}\nAssistant:"
    
    # Step 3: Send to Gemini
    response = model.generate_content(context)
    assistant_message = response.text
    
    # Step 4: Store in history
    conversation_history.append({
        "user": user_message,
        "assistant": assistant_message
    })
    
    # Step 5: Return response
    return assistant_message

# INTERACTIVE CHAT LOOP
print("=== TaskFlow Support Chatbot ===")
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