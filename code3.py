import tkinter as tk
from newspaper import Article
import nltk
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
from rouge_score import rouge_scorer

# Download NLTK punkt
nltk.download('punkt')

# ---------------- Load Models ---------------- #
# BART model for summarization
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Transformer sentiment analysis
sentiment_analyzer = pipeline("sentiment-analysis")

# ---------------- Helper Functions ---------------- #
def evaluate_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores

def get_sentiment(text):
    # Limit to 512 tokens for speed
    result = sentiment_analyzer(text[:512])[0]
    return f"Sentiment: {result['label']}, Score: {result['score']:.2f}"

# ---------------- Summarization Function ---------------- #
def summarize():
    url = utext.get('1.0', "end").strip()
    if not url:
        return

    article = Article(url)
    article.download()
    article.parse()
    article.nlp()

    # Newspaper basic summary
    basic_summary = article.summary

    # Transformer (BART) summary
    inputs = bart_tokenizer([article.text], return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = bart_model.generate(
        inputs['input_ids'], max_length=150, min_length=40,
        length_penalty=2.0, num_beams=4, early_stopping=True
    )
    transformer_summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Sentiment analysis
    sentiment_result = get_sentiment(article.text)

    # Update GUI fields
    for widget, value in zip(
        [title, author, publication, summary_basic, summary_transformer, sentiment],
        [article.title, str(article.authors), str(article.publish_date), basic_summary, transformer_summary, sentiment_result]
    ):
        widget.config(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('1.0', value)
        widget.config(state='disabled')

    # ROUGE evaluation (prints in console)
    rouge_summary = evaluate_rouge(basic_summary, transformer_summary)
    print("\n--- ROUGE for Summaries ---")
    print(rouge_summary)

# ---------------- GUI Setup ---------------- #
root = tk.Tk()
root.title("📰 AI News Summarizer & Sentiment Analyzer")
root.geometry('1200x800')
root.configure(bg="#F3F4F6")
root.resizable(True, True)

# --- Scrollable Frame Setup --- #
main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True)

canvas = tk.Canvas(main_frame, bg="#F3F4F6")
scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# --- Main Heading --- #
heading_label = tk.Label(
    scrollable_frame,
    text="🧠 AI News Summarizer & Sentiment Analyzer 📰",
    font=("Helvetica", 20, "bold"),
    bg="#1E3A8A",
    fg="white",
    padx=20,
    pady=15
)
heading_label.pack(fill="x", pady=(0, 15))

# --- Style for Labels --- #
def create_text_label(parent, label_text, height, bg_color):
    tk.Label(parent, text=label_text, font=("Arial", 12, "bold"), bg="#F3F4F6").pack(pady=(10, 0))
    text_widget = tk.Text(parent, height=height, width=140, bg=bg_color, font=("Consolas", 10))
    text_widget.config(state='disabled', wrap="word")
    text_widget.pack(pady=(0, 5))
    return text_widget

# --- Widgets --- #
title = create_text_label(scrollable_frame, '📰 Title', 1, 'lightyellow')
author = create_text_label(scrollable_frame, '✍️ Author', 1, 'lightyellow')
publication = create_text_label(scrollable_frame, '📅 Publishing Date', 1, 'lightyellow')
summary_basic = create_text_label(scrollable_frame, '🧾 Basic Summary (Newspaper3k)', 10, 'lightblue')
summary_transformer = create_text_label(scrollable_frame, '🤖 Transformer Summary (BART)', 10, 'lightgreen')
sentiment = create_text_label(scrollable_frame, '💭 Sentiment Analysis', 1, 'lightpink')

# --- URL and Button --- #
tk.Label(scrollable_frame, text='🔗 Enter News URL', font=("Arial", 12, "bold"), bg="#F3F4F6").pack(pady=(10, 0))
utext = tk.Text(scrollable_frame, height=1, width=140, font=("Consolas", 10))
utext.insert("1.0", "https://images.dawn.com/news/1194100/james-gunns-superman-sequel-man-of-tomorrow-set-to-release-in-july-2027")
utext.pack(pady=(0, 10))

btn = tk.Button(
    scrollable_frame,
    text="✨ Summarize",
    command=summarize,
    bg="#2563EB",
    fg="white",
    font=("Arial", 12, "bold"),
    relief="raised",
    padx=10,
    pady=5
)
btn.pack(pady=10)

root.mainloop()



#https://www.dawn.com/news/1940759/israel-attacks-hamas-leadership-gathered-in-doha-several-explosions-reported
#https://www.dawn.com/news/1940759/israel-attacks-hamas-leadership-gathered-in-doha-several-explosions-reported