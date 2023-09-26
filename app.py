from flask import Flask, render_template, request
from transformers import BartForConditionalGeneration, BartTokenizer

app = Flask(__name__)

# Load BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def generate_summary(text):
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    
    # Generate summary using BART
    summary_ids = model.generate(
        input_ids,
        max_length=150,         # Adjust the max length as needed
        min_length=50,          # Adjust the min length as needed
        length_penalty=2.0,     # Adjust the length penalty as needed
        num_return_sequences=1,
        no_repeat_ngram_size=2  # Adjust the no_repeat_ngram_size as needed
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    input_text = request.form['input_text']
    summary = generate_summary(input_text)
    return render_template('index.html', input_text=input_text, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
