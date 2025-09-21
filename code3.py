import tkinter as tk
from newspaper import Article
import nltk

# Transformers for summarization and sentiment
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
from rouge_score import rouge_scorer

# Download punkt for newspaper NLP
nltk.download('punkt')

# ---------------- Load Models ---------------- #
# BART model for summarization
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# T5 model for headline generation
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Transformer sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# ---------------- Helper Functions ---------------- #
def generate_headline(article_text, article_title=""):
    # Preprocess text: remove newlines and extra spaces
    article_text = article_text.replace("\n", " ").strip()
    
    # Use first 300 words for context instead of first 5 sentences
    words = article_text.split()
    short_input = " ".join(words[:300])

    # Include article title prominently
    input_text = f"headline: {article_title}. Article: {short_input}"

    # Encode input
    inputs = t5_tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    # Generate headline
    summary_ids = t5_model.generate(
        inputs,
        max_length=25,       # max tokens in headline
        min_length=8,        # minimum tokens to avoid very short headlines
        num_beams=8,         # beam search for better quality
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    headline = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return headline.strip()

def evaluate_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores



def get_sentiment(text):
    # Limit to 512 tokens for speed
    result = sentiment_analyzer(text[:512])[0]
    return f"Sentiment: {result['label']}, Score: {result['score']:.2f}"

# ---------------- Main Summarization Function ---------------- #
def summarize():
    url = utext.get('1.0', "end").strip()
    if not url:
        return

    # Fetch article
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()

    # Newspaper summary
    basic_summary = article.summary

    # Transformer summarization (BART)
    inputs = bart_tokenizer([article.text], return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = bart_model.generate(
        inputs['input_ids'], max_length=150, min_length=40,
        length_penalty=2.0, num_beams=4, early_stopping=True
    )
    transformer_summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Headline generation (T5)
    headline = generate_headline(article.text, article.title)

    # Transformer-based sentiment
    sentiment_result = get_sentiment(article.text)

    # ---------------- Display in GUI ---------------- #
    for widget, value in zip(
        [title, author, publication, summary_basic, summary_transformer, headline_box, sentiment],
        [article.title, str(article.authors), str(article.publish_date), basic_summary, transformer_summary, headline, sentiment_result]
    ):
        widget.config(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('1.0', value)
        widget.config(state='disabled')

    # ---------------- ROUGE Evaluation ---------------- #
    rouge_headline = evaluate_rouge(article.title, headline)
    rouge_summary = evaluate_rouge(basic_summary, transformer_summary)

    print("\n--- ROUGE for Headline ---")
    print(rouge_headline)

    print("\n--- ROUGE for Summaries ---")
    print(rouge_summary)

# ---------------- GUI Setup ---------------- #
root = tk.Tk()
root.title("News Summarizer + Headline Generator")
root.geometry('1200x800')

# GUI Widgets
def create_text_label(label_text, height, bg_color):
    tk.Label(root, text=label_text).pack()
    text_widget = tk.Text(root, height=height, width=140, bg=bg_color)
    text_widget.config(state='disabled')
    text_widget.pack()
    
    return text_widget

title = create_text_label('Title', 1, 'lightyellow')
author = create_text_label('Author', 1, 'lightyellow')
publication = create_text_label('Publishing Date', 1, 'lightyellow')
summary_basic = create_text_label('Basic Summary (Newspaper3k)', 10, 'lightblue')
summary_transformer = create_text_label('Transformer Summary (BART)', 10, 'lightgreen')
headline_box = create_text_label('Generated Headline (T5)', 2, 'lightcoral')
sentiment = create_text_label('Sentiment Analysis', 1, 'lightpink')

tk.Label(root, text='URL').pack()
utext = tk.Text(root, height=1, width=140)
utext.insert("1.0", "https://images.dawn.com/news/1194100/james-gunns-superman-sequel-man-of-tomorrow-set-to-release-in-july-2027")
utext.pack()

btn = tk.Button(root, text="Summarize", command=summarize)
btn.pack()

root.mainloop()
#https://www.dawn.com/news/1940759/israel-attacks-hamas-leadership-gathered-in-doha-several-explosions-reported
#https://www.dawn.com/news/1940759/israel-attacks-hamas-leadership-gathered-in-doha-several-explosions-reported