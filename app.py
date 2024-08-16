from flask import Flask, request, jsonify, render_template
from langchain_community.llms import Ollama
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnableSequence
import re

app = Flask(__name__)

# Initialize Ollama LLM
llm = Ollama(model="llama3")

# Enhanced prompt templates for generating comprehensive arguments
for_prompt = PromptTemplate.from_template(
    "As a skilled debater, write a detailed multi-paragraph argument in favor of the statement: '{statement}'. "
    "Start with a strong introduction, followed by three well-supported points, each with evidence or examples, "
    "and finish with a compelling conclusion that reinforces your stance."
)

against_prompt = PromptTemplate.from_template(
    "As a skilled debater, write a detailed multi-paragraph argument against the statement: '{statement}'. "
    "Start with a strong introduction, followed by three well-supported points, each with evidence or examples, "
    "and finish with a compelling conclusion that reinforces your stance."
)

# Create RunnableSequences for argument generation
for_chain = RunnableSequence(for_prompt, llm)
against_chain = RunnableSequence(against_prompt, llm)

# Function to check for biased or harmful content
def content_filter(text):
    if re.search(r'\b(bias|harm|sensitive)\b', text, re.IGNORECASE):
        return "Filtered due to sensitive content"
    return text

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        statement = request.form.get('statement')
        for_argument = content_filter(for_chain.invoke({"statement": statement}))
        against_argument = content_filter(against_chain.invoke({"statement": statement}))
        return render_template('index.html', for_argument=for_argument, against_argument=against_argument, statement=statement)
    return render_template('index.html')

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

@app.route('/feedback', methods=['POST'])
def feedback():
    for_argument = request.form.get('for_argument')
    against_argument = request.form.get('against_argument')
    rating = int(request.form.get('rating'))
    
    # Store feedback (This can be extended to save to a database)
    print(f"Feedback received: For: {for_argument}, Against: {against_argument}, Rating: {rating}")
    
    return jsonify({"message": "Feedback received"}), 200

if __name__ == '__main__':
    app.run(debug=True)
