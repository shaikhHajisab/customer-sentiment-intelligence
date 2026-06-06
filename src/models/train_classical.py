import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import sys

# Add src to path to import features module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features.build_features import get_tfidf_vectorizer

def train_and_evaluate():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../reports'))
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    print("Loading datasets...")
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'validation.csv'))
    
    # Fill any NaNs that slipped through
    train_df['clean_text'] = train_df['clean_text'].fillna('')
    val_df['clean_text'] = val_df['clean_text'].fillna('')
    
    print("Extracting TF-IDF features...")
    vectorizer = get_tfidf_vectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['clean_text'])
    X_val = vectorizer.transform(val_df['clean_text'])
    
    # Save vectorizer
    with open(os.path.join(models_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
        
    y_train = train_df['label']
    y_val = val_df['label']
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, n_jobs=-1),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        
        acc = accuracy_score(y_val, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, preds, average='weighted')
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
        
        # Save model
        with open(os.path.join(models_dir, f'{name.replace(" ", "_").lower()}.pkl'), 'wb') as f:
            pickle.dump(model, f)
            
        # Plot confusion matrix
        cm = confusion_matrix(y_val, preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['negative', 'neutral', 'positive'], 
                    yticklabels=['negative', 'neutral', 'positive'])
        plt.title(f'{name} Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(reports_dir, f'cm_{name.replace(" ", "_").lower()}.png'))
        plt.close()
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(reports_dir, 'model_comparison.csv'), index=False)
    
    print("\nModel Comparison:")
    print(results_df.to_string(index=False))
    print("\nTraining completed successfully.")

if __name__ == "__main__":
    train_and_evaluate()
