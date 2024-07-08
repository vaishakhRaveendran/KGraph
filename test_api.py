import os
import requests

# Load API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    print("API key not found in environment variables.")
    exit()

api_url = 'https://api.openai.com/v1/engines/gpt-3.5-turbo-instruct/completions'

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}',
}

payload = {
    'prompt': 'Once upon a time',
    'max_tokens': 50
}

try:
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        print("API key is working.")
        print("Response:")
        print(response.json())
    else:
        print(f"Error: {response.status_code} - {response.text}")
except Exception as e:
    print(f"Error connecting to API: {e}")
