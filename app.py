from flask import Flask, render_template, request, jsonify
from chatbot import greeting, get_response as get_local_response
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_text = request.json['message']
    
    # Check for exit command
    if user_text.lower() in ['bye', 'goodbye', 'exit', 'quit']:
        return jsonify({'response': 'Goodbye! Have a great day!', 'type': 'local'})
    
    # Check for greeting
    greet = greeting(user_text)
    if greet is not None:
        return jsonify({'response': greet, 'type': 'local'})
    
    try:
        # Try to get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Keep your responses concise and to the point."},
                {"role": "user", "content": user_text}
            ],
            max_tokens=150,
            temperature=0.7
        )
        bot_response = response.choices[0].message.content.strip()
        return jsonify({'response': bot_response, 'type': 'ai'})
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        # Fallback to local response if API fails
        bot_response = get_local_response(user_text)
        return jsonify({'response': bot_response, 'type': 'local'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
