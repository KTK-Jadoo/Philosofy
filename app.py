from flask import Flask, request, jsonify, render_template
from langchain_community.llms import Ollama
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnableSequence
import psycopg2
import re
import os

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
    conn = psycopg2.connect(
        dbname="philosify_feedbacks_db",
        user="Jadoo",        # Replace with your PostgreSQL username
        password=os.getenv("POSTGRESQL_PASSWORD"), # Replace with your PostgreSQL password
        host="localhost",
        port="5432"
    )
    return conn

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
    
    # Store feedback in PostgreSQL database
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO feedback (for_argument, against_argument, rating) VALUES (%s, %s, %s)",
        (for_argument, against_argument, rating)
    )
    conn.commit()
    cur.close()
    conn.close()
    
    return jsonify({"message": "Feedback received"}), 200

if __name__ == '__main__':
    app.run(debug=True)
