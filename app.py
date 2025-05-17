import pandas as pd
import re
import os
from transformers import pipeline
import gradio as gr

# Initialize the emotion detection model
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# Function to load dataset from local file
def load_local_dataset():
    try:
        csv_file = "dataset/sentimentdataset.csv"
        if not os.path.exists(csv_file):
            return "Error: 'sentimentdataset.csv' not found in the 'dataset/' folder."
        
        df = pd.read_csv(csv_file)
        
        # Ensure the dataset has a 'Text' column (case-sensitive)
        if 'Text' not in df.columns:
            return "Error: Dataset must contain a 'Text' column."
        
        # Rename 'Text' to 'text' for consistency
        df = df.rename(columns={'Text': 'text'})
        
        return df
    except Exception as e:
        return f"Error loading dataset: {str(e)}"

# Function to clean text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)    # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text) # Remove special characters
    text = text.lower().strip()
    return text

# Function to analyze emotion for a single text
def analyze_emotion(text):
    if not text or not isinstance(text, str):
        return "Error: Please provide valid text input."
    
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return "Error: Text is empty after cleaning."
    
    try:
        results = classifier(cleaned_text)[0]
        # Format results as a string
        output = "Emotion Analysis Results:\n"
        for result in results:
            label = result['label'].capitalize()
            score = result['score']
            output += f"{label}: {score:.2%}\n"
        return output
    except Exception as e:
        return f"Error during analysis: {str(e)}"

# Function to analyze emotions in a CSV or TXT file
def analyze_file(file):
    try:
        # Check file extension
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
            if 'Text' in df.columns:
                df = df.rename(columns={'Text': 'text'})
            if 'text' not in df.columns:
                return "Error: CSV must contain a 'text' or 'Text' column."
        elif file.name.endswith('.txt'):
            with open(file.name, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            df = pd.DataFrame([line.strip() for line in lines if line.strip()], columns=['text'])
        else:
            return "Error: File must be a CSV or TXT file."
        
        if df.empty:
            return "Error: File is empty."
        
        # Clean text and analyze emotions
        df['cleaned_text'] = df['text'].apply(clean_text)
        results = []
        for text in df['cleaned_text']:
            if text:
                result = classifier(text)[0]
                # Get the dominant emotion
                dominant = max(result, key=lambda x: x['score'])
                results.append(f"{dominant['label'].capitalize()}: {dominant['score']:.2%}")
            else:
                results.append("Skipped: Empty text")
        
        # Add results to dataframe
        df['emotion'] = results
        output = "File Analysis Results:\n\n"
        for idx, row in df.iterrows():
            output += f"Text: {row['text']}\nEmotion: {row['emotion']}\n\n"
        return output
    except Exception as e:
        return f"Error processing file: {str(e)}"

# Function to load and display sample dataset
def load_sample_dataset():
    try:
        df = load_local_dataset()
        if isinstance(df, str):  # Error message
            return df
        
        # Limit to first 5 rows for display
        output = "Sample Dataset Preview (First 5 Rows):\n\n"
        for idx, row in df.head().iterrows():
            output += f"Text: {row['text']}\n"
            # Check if there's a 'Sentiment' column
            if 'Sentiment' in df.columns:
                output += f"Labeled Sentiment: {row['Sentiment']}\n"
            output += "\n"
        return output
    except Exception as e:
        return f"Error loading dataset: {str(e)}"

# Define a custom Gradio theme for a modern look
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    radius_size="lg",
    text_size="md",
)

# Gradio interface with card-based layout
with gr.Blocks(title="Emotion Detection in Social Media Conversations", theme=theme) as demo:
    gr.Markdown(
        """
        # Emotion Detection in Social Media Conversations 💬
        Uncover the emotions behind social media posts using advanced sentiment analysis.  
        Powered by a pre-trained Hugging Face model.  
        """
    )

    # Card 1: Dataset Preview
    with gr.Group():
        gr.Markdown("### 📊 Explore the Social Media Dataset")
        gr.Markdown(
            "Preview the dataset to see sample posts and their labeled sentiments.",
            elem_classes="description-text"
        )
        with gr.Row():
            dataset_button = gr.Button("Load Dataset Preview", variant="primary")
            dataset_output = gr.Textbox(
                label="",
                placeholder="Click 'Load Dataset Preview' to see the data...",
                lines=8,
                max_lines=15,
                show_label=False,
                elem_classes="output-box"
            )
        dataset_button.click(fn=load_sample_dataset, outputs=dataset_output)

    # Card 2: Single Text Analysis
    with gr.Group():
        gr.Markdown("### ✍️ Analyze a Single Post")
        gr.Markdown(
            "Enter a social media post to detect its underlying emotions.",
            elem_classes="description-text"
        )
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="",
                    placeholder="Type a post, e.g., 'Feeling on top of the world today! 🌍'",
                    lines=2,
                    show_label=False,
                    elem_classes="input-box"
                )
            with gr.Column(scale=1):
                text_button = gr.Button("Detect Emotions", variant="primary")
        text_output = gr.Textbox(
            label="",
            placeholder="Emotion results will appear here...",
            lines=5,
            max_lines=10,
            show_label=False,
            elem_classes="output-box"
        )
        text_button.click(fn=analyze_emotion, inputs=text_input, outputs=text_output)

    # Card 3: Batch File Analysis
    with gr.Group():
        gr.Markdown("### 📁 Analyze Multiple Posts")
        gr.Markdown(
            "Upload a CSV or TXT file to analyze emotions in multiple posts at once. CSV files must have a 'text' or 'Text' column; TXT files should have one post per line.",
            elem_classes="description-text"
        )
        with gr.Row():
            file_input = gr.File(
                label="",
                file_types=[".csv", ".txt"],
                show_label=False,
                elem_classes="input-box"
            )
            file_button = gr.Button("Analyze File", variant="primary")
        file_output = gr.Textbox(
            label="",
            placeholder="Batch analysis results will appear here...",
            lines=8,
            max_lines=20,
            show_label=False,
            elem_classes="output-box"
        )
        file_button.click(fn=analyze_file, inputs=file_input, outputs=file_output)

    # Footer
    gr.Markdown(
        """
        ---
        Built with ❤️ using Gradio and Hugging Face.  
        Dataset: Social Media Sentiments Analysis Dataset  
        Model: `j-hartmann/emotion-english-distilroberta-base`  
        """
    )

# Launch the app (commented out for Hugging Face Spaces)
# demo.launch()

if __name__ == "__main__":
    demo.launch()
