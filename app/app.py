import streamlit as st
import requests
import pandas as pd
import os
from PIL import Image

# Configuration
API_URL = "http://localhost:8000/predict"
st.set_page_config(page_title="Customer Sentiment Intelligence", page_icon="📊", layout="wide")

# Initialize session state for prediction history
if 'history' not in st.session_state:
    st.session_state.history = []

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Real-time Prediction", "Model Analytics"])

    if page == "Real-time Prediction":
        show_prediction_page()
    elif page == "Model Analytics":
        show_analytics_page()

def show_prediction_page():
    st.title("🗣️ Real-time Sentiment Prediction")
    st.markdown("Enter a customer review or tweet below to analyze its sentiment using our fine-tuned RoBERTa model.")

    user_input = st.text_area("Enter Text:", height=150, placeholder="E.g., The customer service was absolutely fantastic!")

    if st.button("Analyze Sentiment"):
        if not user_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(API_URL, json={"text": user_input})
                    if response.status_code == 200:
                        result = response.json()
                        display_result(result)
                        # Add to history
                        st.session_state.history.append({
                            "Text": result['text'],
                            "Sentiment": result['label'].capitalize(),
                            "Confidence": f"{result['confidence']:.2%}"
                        })
                    else:
                        st.error(f"API Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to the backend API. Ensure the FastAPI server is running on localhost:8000.")

    # Show History
    if st.session_state.history:
        st.markdown("---")
        st.subheader("Prediction History")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

def display_result(result):
    label = result['label']
    conf = result['confidence']
    
    # Set colors based on sentiment
    if label == "positive":
        color = "green"
        emoji = "😊"
    elif label == "negative":
        color = "red"
        emoji = "😞"
    else:
        color = "gray"
        emoji = "😐"
        
    st.markdown(f"### Result: <span style='color:{color}'>{label.capitalize()} {emoji}</span>", unsafe_allow_html=True)
    st.progress(conf)
    st.write(f"**Confidence Score:** {conf:.2%}")

def show_analytics_page():
    st.title("📈 Model Analytics & EDA Insights")
    st.markdown("Compare the performance of our classical Machine Learning models against the dataset.")

    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../reports'))
    
    # Display comparison metrics
    csv_path = os.path.join(reports_dir, 'model_comparison.csv')
    if os.path.exists(csv_path):
        st.subheader("Classical Models Performance")
        df = pd.read_csv(csv_path)
        st.dataframe(df.style.highlight_max(axis=0, color='lightgreen', subset=['Accuracy', 'F1 Score']), use_container_width=True)
        
        # Simple bar chart
        st.bar_chart(df.set_index('Model')['Accuracy'])
    else:
        st.info("Model comparison data not found. Run the training script first.")

    st.markdown("---")
    st.subheader("Exploratory Data Analysis (EDA)")
    
    col1, col2 = st.columns(2)
    
    def display_image(filename, caption, col):
        img_path = os.path.join(reports_dir, filename)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            col.image(img, caption=caption, use_container_width=True)

    display_image('class_distribution.png', 'Sentiment Class Distribution', col1)
    display_image('text_length_dist.png', 'Text Length Distribution', col2)
    display_image('wordcloud_positive.png', 'Positive Word Cloud', col1)
    display_image('wordcloud_negative.png', 'Negative Word Cloud', col2)

    st.markdown("---")
    st.subheader("Confusion Matrices")
    
    cm_cols = st.columns(4)
    models = ['logistic_regression', 'naive_bayes', 'random_forest', 'xgboost']
    
    for idx, model_name in enumerate(models):
        img_path = os.path.join(reports_dir, f'cm_{model_name}.png')
        if os.path.exists(img_path):
            img = Image.open(img_path)
            cm_cols[idx].image(img, caption=model_name.replace("_", " ").title(), use_container_width=True)

if __name__ == "__main__":
    main()
