from dotenv import load_dotenv
from google import genai

load_dotenv()  # Loads the .env file automatically

# Creating a Client automatically fetches your GEMINI_API_KEY environment variable!
client = genai.Client()

# Ask the upgraded 2.5 engine directly
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Does TaskFlow integrate with Jira?",
)

print(response.text)