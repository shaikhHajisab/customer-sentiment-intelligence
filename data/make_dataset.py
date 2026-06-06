import os
import re
import pandas as pd
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def clean_text(text):
    text = str(text).lower()
    # Remove mentions
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

def process_and_save():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading tweet_eval sentiment dataset...")
    dataset = load_dataset("tweet_eval", "sentiment")
    
    label_mapping = {0: "negative", 1: "neutral", 2: "positive"}
    
    for split in ["train", "validation", "test"]:
        df = dataset[split].to_pandas()
        print(f"Processing {split} split ({len(df)} samples)...")
        
        df['clean_text'] = df['text'].apply(clean_text)
        df['label_text'] = df['label'].map(label_mapping)
        
        # Remove empty rows after cleaning
        df = df[df['clean_text'].str.strip().astype(bool)]
        
        output_path = os.path.join(output_dir, f"{split}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {split} to {output_path}")

if __name__ == "__main__":
    process_and_save()
