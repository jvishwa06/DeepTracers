# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from langchain_ollama import OllamaLLM

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains
llm = OllamaLLM(model="qwen2.5:0.5b")  # Initialize the LLM

@app.route('/api/chat', methods=['POST'])  # Define the endpoint to accept POST requests
def chat():
    user_input = request.json.get('input', '')  # Get user input from the request
    response = llm.invoke(input=user_input)  # Invoke the LLM with the user input
    return jsonify({'response': response})  # Return the response as JSON

if __name__ == '__main__':
    app.run(port=5002)  # Start the Flask app on port 5002
