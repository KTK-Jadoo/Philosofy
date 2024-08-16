from flask import Flask, request, jsonify, render_template
from langchain_community.llms import Ollama
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnableSequence
import psycopg2
import re
import os
import sqlite3


app = Flask(__name__)

# Initialize Ollama LLM
llm = Ollama(model="llama3")

# Refined prompt templates for generating concise, philosophical arguments
for_prompt = PromptTemplate.from_template(
    "In no more than 115 words, explain why the following statement is true: '{statement}'. "
    "Keep your argument philosophical, direct, and conversational."
)

against_prompt = PromptTemplate.from_template(
    "In no more than 115 words, explain why the following statement is false: '{statement}'. "
    "Keep your argument philosophical, direct, and conversational."
)

# Create RunnableSequences for argument generation
for_chain = RunnableSequence(for_prompt, llm)
against_chain = RunnableSequence(against_prompt, llm)

# Database connection setup
def get_db_connection():
    conn = sqlite3.connect('database.db')  # This will create a file named 'database.db'
    conn.row_factory = sqlite3.Row         # This allows accessing columns by name
    return conn


def init_db():
    conn = get_db_connection()
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                for_argument TEXT NOT NULL,
                against_argument TEXT NOT NULL,
                rating INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    conn.close()


# Function to check for biased or harmful content
def content_filter(text):
    if re.search(r'\b(bias|harm|sensitive)\b', text, re.IGNORECASE):
        return "Filtered due to sensitive content"
    return text

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        statement = request.form.get('statement')
        if not statement:
            return render_template('index.html', error="Please enter a statement.")
        for_argument = content_filter(for_chain.invoke({"statement": statement}))
        against_argument = content_filter(against_chain.invoke({"statement": statement}))
        return render_template('index.html', for_argument=for_argument, against_argument=against_argument, statement=statement)
    return render_template('index.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    for_argument = request.form.get('for_argument')
    against_argument = request.form.get('against_argument')
    rating = int(request.form.get('rating'))
    
    conn = get_db_connection()
    with conn:
        conn.execute(
            'INSERT INTO feedback (for_argument, against_argument, rating) VALUES (?, ?, ?)',
            (for_argument, against_argument, rating)
        )
    
    return jsonify({"message": "Feedback received"}), 200

@app.route('/continue_debate', methods=['POST'])
def continue_debate():
    data = request.json
    previous_statement = data.get('previous_statement')
    user_response = data.get('user_response')
    
    if not previous_statement or not user_response:
        return jsonify({"error": "No statement or response provided"}), 400
    
    # Generate a counterargument based on the user's response
    counter_argument = content_filter(against_chain.invoke({"statement": user_response}))
    
    return jsonify({
        "previous_statement": previous_statement,
        "user_response": user_response,
        "counter_argument": counter_argument
    })


if __name__ == '__main__':
    init_db()  # Initialize the database with the feedback table
    app.run(debug=True)

