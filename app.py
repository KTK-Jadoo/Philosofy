from flask import Flask, request, jsonify, render_template_string
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re

app = Flask(__name__)

# Initialize FLAN-T5 model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create a text-generation pipeline
pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=512
)

# Create LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Create prompt templates for generating arguments
for_prompt = PromptTemplate(
    input_variables=["statement"],
    template="Generate a logically sound argument in favor of the following statement: {statement}"
)

against_prompt = PromptTemplate(
    input_variables=["statement"],
    template="Generate a logically sound argument against the following statement: {statement}"
)

# Logical consistency prompt (hypothetical example)
consistency_prompt = PromptTemplate(
    input_variables=["for_argument", "against_argument"],
    template="Evaluate the logical consistency between the following arguments. For: {for_argument}. Against: {against_argument}."
)

# Create LLMChains
for_chain = LLMChain(llm=llm, prompt=for_prompt)
against_chain = LLMChain(llm=llm, prompt=against_prompt)

# HTML template
HTML = '''
<!doctype html>
<html>
    <head>
        <title>Argument Generator</title>
    </head>
    <body>
        <h1>Argument Generator</h1>
        <form method="POST">
            <textarea name="statement" rows="4" cols="50"></textarea><br>
            <input type="submit" value="Generate Arguments">
        </form>
        {% if for_argument and against_argument %}
        <h2>Results:</h2>
        <h3>Argument For:</h3>
        <p>{{ for_argument }}</p>
        <h3>Argument Against:</h3>
        <p>{{ against_argument }}</p>
        <h3>Logical Consistency Score:</h3>
        <p>{{ consistency_score }}</p>
        <form method="POST" action="/feedback">
            <input type="hidden" name="for_argument" value="{{ for_argument }}">
            <input type="hidden" name="against_argument" value="{{ against_argument }}">
            <label for="rating">Rate the arguments (1-5):</label>
            <input type="number" name="rating" min="1" max="5"><br>
            <input type="submit" value="Submit Feedback">
        </form>
        {% endif %}
    </body>
</html>
'''

# Function to check for biased or harmful content
def content_filter(text):
    if re.search(r'\b(bias|harm|sensitive)\b', text, re.IGNORECASE):
        return "Filtered due to sensitive content"
    return text

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        statement = request.form.get('statement')
        for_argument = content_filter(for_chain.run(statement))
        against_argument = content_filter(against_chain.run(statement))
        consistency_score = evaluate_consistency(for_argument, against_argument)
        return render_template_string(HTML, for_argument=for_argument, against_argument=against_argument, consistency_score=consistency_score)
    return render_template_string(HTML)

@app.route('/generate_arguments', methods=['POST'])
def generate_arguments():
    data = request.json
    statement = data.get('statement')
    
    if not statement:
        return jsonify({"error": "No statement provided"}), 400
    
    # Generate arguments
    for_argument = content_filter(for_chain.run(statement))
    against_argument = content_filter(against_chain.run(statement))
    consistency_score = evaluate_consistency(for_argument, against_argument)
    
    return jsonify({
        "for_argument": for_argument,
        "against_argument": against_argument,
        "consistency_score": consistency_score
    })

# Placeholder function for evaluating logical consistency
def evaluate_consistency(for_argument, against_argument):
    consistency_response = pipe({
        "inputs": consistency_prompt.format(for_argument=for_argument, against_argument=against_argument)
    })
    # Hypothetical score extraction
    score_match = re.search(r'(\d+)/10', consistency_response[0]['generated_text'])
    return score_match.group(0) if score_match else "Not evaluated"

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
