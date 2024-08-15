from flask import Flask, request, jsonify, render_template
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import RunnableSequence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re

app = Flask(__name__)

# Initialize FLAN-T5 model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create a text-generation pipeline with enhanced parameters
pipe = pipeline(
    "text2text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=2048,  # Increased max_length for more detailed and multi-paragraph responses
    temperature=0.65,  # Slightly reduced temperature for more focused arguments
    top_p=0.85,        # Nucleus sampling with a lower top_p for more coherent outputs
    do_sample=True,    # Enable sampling to allow temperature and top_p to take effect
    repetition_penalty=1.2  # Added repetition penalty to reduce redundancy
)

# Create LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Enhanced prompt templates for generating comprehensive arguments
for_prompt = PromptTemplate.from_template(
    "You are a seasoned lawyer presenting a multi-paragraph argument in favor of the statement: '{statement}'. "
    "Ensure that your argument includes an introduction, several key points with evidence, and a conclusion."
)

against_prompt = PromptTemplate.from_template(
    "You are a seasoned lawyer presenting a multi-paragraph argument against the statement: '{statement}'. "
    "Ensure that your argument includes an introduction, several key points with evidence, and a conclusion."
)

# Create a prompt template for evaluating logical consistency
consistency_prompt = PromptTemplate.from_template(
    "Evaluate the logical consistency between the following arguments. For: {for_argument}. Against: {against_argument}."
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
        consistency_score = evaluate_consistency(for_argument, against_argument)
        return render_template('index.html', for_argument=for_argument, against_argument=against_argument, consistency_score=consistency_score)
    return render_template('index.html')

@app.route('/generate_arguments', methods=['POST'])
def generate_arguments():
    data = request.json
    statement = data.get('statement')
    
    if not statement:
        return jsonify({"error": "No statement provided"}), 400
    
    # Generate arguments
    for_argument = content_filter(for_chain.invoke({"statement": statement}))
    against_argument = content_filter(against_chain.invoke({"statement": statement}))
    consistency_score = evaluate_consistency(for_argument, against_argument)
    
    return jsonify({
        "for_argument": for_argument,
        "against_argument": against_argument,
        "consistency_score": consistency_score
    })

# Updated function for evaluating logical consistency
def evaluate_consistency(for_argument, against_argument):
    prompt = consistency_prompt.format(for_argument=for_argument, against_argument=against_argument)
    consistency_response = pipe(prompt)  # Pass the prompt as a string
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
