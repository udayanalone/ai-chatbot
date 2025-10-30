from dotenv import load_dotenv
import os
from openai import OpenAI

def test_openai_connection():
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return False
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Try a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello!"}
            ],
            max_tokens=50
        )
        
        # Print response
        print("OpenAI API Test Response:", response.choices[0].message.content)
        return True
        
    except Exception as e:
        print(f"Error testing OpenAI connection: {e}")
        return False

if __name__ == "__main__":
    success = test_openai_connection()
    print(f"\nTest {'succeeded' if success else 'failed'}")