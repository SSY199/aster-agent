from google import genai

# Automatically detects your GEMINI_API_KEY environment variable
client = genai.Client()

print("Available Gemini Models:")
for model in client.models.list():
    print(model.name)
