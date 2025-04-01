from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
import re
import base64
import io
from flask_session import Session
import hashlib
import os
import traceback
from werkzeug.middleware.profiler import ProfilerMiddleware

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_session'
app.config['SESSION_FILE_THRESHOLD'] = 100
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

Session(app)

# Configure Gemini
def configure_gemini():
    genai.configure(api_key="AIzaSyDYNsbwVVVYlj0Szr15ZEGO-Eb8F-bI7Jc")

def get_gemini_model():
    try:
        configure_gemini()
        return genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 2048,
            },
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_DANGEROUS",
                    "threshold": "BLOCK_NONE",
                },
            ]
        )
    except Exception as e:
        app.logger.error(f"Gemini initialization failed: {str(e)}")
        raise

def enhanced_text_analysis(text):
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    char_count = len(text)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    total_word_length = sum(len(word) for word in words)
    avg_word_len = round(total_word_length / word_count, 1) if word_count > 0 else 0
    
    word_counts = Counter(words)
    common_words = dict(word_counts.most_common(10))
    
    positive_words = ['good', 'great', 'excellent', 'positive', 'happy', 'love', 'best', 'wonderful', 'amazing', 'fantastic']
    negative_words = ['bad', 'poor', 'negative', 'sad', 'hate', 'terrible', 'worst', 'awful', 'horrible', 'disappointing']
    
    positive_count = sum(words.count(word) for word in positive_words)
    negative_count = sum(words.count(word) for word in negative_words)
    neutral_count = word_count - (positive_count + negative_count)
    
    sentiment_score = (positive_count - negative_count) / (word_count) * 10 if word_count > 0 else 0
    
    if sentiment_score > 3:
        sentiment = "Positive"
    elif sentiment_score < -3:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'sentence_count': sentence_count,
        'avg_word_len': avg_word_len,
        'common_words': common_words,
        'positive_words': positive_count,
        'negative_words': negative_count,
        'neutral_words': neutral_count,
        'sentiment_score': sentiment_score,
        'sentiment': sentiment
    }

def analyze_text_with_gemini(text):
    try:
        model = get_gemini_model()
        
        prompt = f"""Analyze the following conversation/text comprehensively and provide detailed insights about:
        
        1. For each participant (if a conversation) or the author (if single person):
           - Personality Traits
           - Emotional State
           - Interests/Hobbies
           - Writing/Speaking Style
           - Potential Demographics (age, gender, education level, cultural background)
           - Other Relevant Characteristics
        
        2. Key Themes and Topics discussed
        
        3. Overall Sentiment Analysis
        
        4. Communication Patterns and Dynamics (for conversations)
        
        5. Notable Linguistic Features
        
        Structure your response with clear headings and bullet points. Be as detailed and analytical as possible.
        
        Text to analyze:
        {text}
        
        Analysis:
        """
        
        response = model.generate_content(prompt)
        
        # Format the response with HTML for better presentation
        formatted_response = response.text.replace("\n\n", "<br><br>")
        formatted_response = formatted_response.replace("\n", "<br>")
        formatted_response = formatted_response.replace("**", "<b>").replace("**", "</b>")
        formatted_response = formatted_response.replace("* ", "• ")
        
        return f"""
        <div class="gemini-analysis">
            <h3>🧠 Comprehensive Text Analysis</h3>
            <div class="analysis-content">{formatted_response}</div>
        </div>
        """
    except Exception as e:
        return f"<div class='alert alert-danger'>Error analyzing text: {str(e)}</div>"
def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    img_data = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(img_data, format='png')
    plt.close()
    img_data.seek(0)
    encoded = base64.b64encode(img_data.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

def create_common_words_chart_data(common_words):
    df = pd.DataFrame({
        'word': list(common_words.keys()),
        'count': list(common_words.values())
    })
    df = df.sort_values('count', ascending=False)
    return {
        'labels': df['word'].tolist(),
        'values': df['count'].tolist()
    }

def create_sentiment_chart_data(positive_count, negative_count, neutral_count):
    return {
        'labels': ['Positive', 'Negative', 'Neutral'],
        'values': [positive_count, negative_count, neutral_count],
        'colors': ["#4CAF50", "#F44336", "#9E9E9E"]
    }

def chat_with_data(text, question, chat_history):
    try:
        model = get_gemini_model()
        
        context = ""
        if chat_history:
            for entry in chat_history:
                context += f"User: {entry['user']}\nAI: {entry['ai']}\n\n"
        
        prompt = f"""
        Based on this text: 
        {text}
        
        Previous conversation:
        {context}
        
        Answer this question: {question}
        
        If you cannot answer based on the provided text, say so clearly.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"I'm sorry, I couldn't process your question due to an error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if request.is_json:
            data = request.get_json()
            user_text = data.get('text', '').strip()
        else:
            user_text = request.form.get('text', '').strip()
        
        if not user_text:
            return jsonify({"error": "Please enter some text to analyze"}), 400
        
        if len(user_text) > 10000:
            return jsonify({"error": "Text too long (max 10,000 characters)"}), 400
        
        session['text_hash'] = user_text
        session['chat_history'] = []
        
        basic_stats = enhanced_text_analysis(user_text)
        
        try:
            gemini_response = analyze_text_with_gemini(user_text)
        except Exception as e:
            app.logger.error(f"Gemini analysis failed: {str(e)}")
            gemini_response = "<div class='alert alert-warning'>Analysis incomplete due to API limitations</div>"
        
        wordcloud_img = create_wordcloud(user_text)
        common_words_data = create_common_words_chart_data(basic_stats['common_words'])
        sentiment_data = create_sentiment_chart_data(
            basic_stats['positive_words'], 
            basic_stats['negative_words'],
            basic_stats['neutral_words']
        )
        
        return jsonify({
            "basic_stats": basic_stats,
            "gemini_response": gemini_response,
            "visualizations": {
                "wordcloud": wordcloud_img.split('base64,')[1],
                "common_words": {
                    "words": list(basic_stats['common_words'].keys()),
                    "counts": list(basic_stats['common_words'].values())
                },
                "sentiment": {
                    "labels": ["Positive", "Negative", "Neutral"],
                    "values": [
                        basic_stats['positive_words'],
                        basic_stats['negative_words'],
                        basic_stats['neutral_words']
                    ],
                    "colors": ["#4CAF50", "#F44336", "#9E9E9E"]
                }
            },
            "text_hash": session['text_hash']
        })
        
    except Exception as e:
        app.logger.error(f"Analysis error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if request.is_json:
            data = request.get_json()
            question = data.get('question', '').strip()
            text_hash = data.get('text_hash', '')
        else:
            question = request.form.get('question', '').strip()
            text_hash = request.form.get('text_hash', '')
        
        if not question:
            return jsonify({'error': 'Please enter a question'})
        
        if not text_hash or text_hash != session.get('text_hash'):
            return jsonify({'error': 'No text to analyze or session expired. Please analyze text first.'})
        
        chat_history = session.get('chat_history', [])
        response = chat_with_data(f"Text with hash: {text_hash}", question, chat_history)
        
        if len(chat_history) >= 5:
            chat_history.pop(0)
            
        chat_history.append({
            'user': question,
            'ai': response
        })
        session['chat_history'] = chat_history
        
        return jsonify({
            'response': response,
            'history': chat_history
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'})

if __name__ == '__main__':
    os.makedirs('./flask_session', exist_ok=True)
    app.run(host="0.0.0.0", port=5000)