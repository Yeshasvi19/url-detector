from flask import Flask, request, render_template, redirect, url_for, session, send_from_directory
import sqlite3
import joblib
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your secret key

# Load the model and scaler
model_path = r'D:\URL\malicious_url_detector.pkl'  # Update with your actual path
scaler_path = r'D:\URL\scaler.pkl'  # Update with your actual path
xgb_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Directory to store plot images
PLOT_DIR = 'static/plots'
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# Function to create database if it doesn't exist
def init_db():
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        url TEXT NOT NULL,
                        prediction TEXT NOT NULL,
                        feedback TEXT NOT NULL,
                        comments TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL UNIQUE,
                        password TEXT NOT NULL)''')
    conn.close()

# Initialize the database
init_db()

@app.route('/', methods=['GET'])
def signuporlogin():
    if 'user_id' in session:
        return redirect(url_for('index'))
    return render_template('signuporlogin.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        
        conn = sqlite3.connect('feedback.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('feedback.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            return redirect(url_for('index'))
        else:
            return 'Invalid username or password'
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('signuporlogin'))

@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'user_id' not in session:
        return redirect(url_for('signuporlogin'))
    
    if request.method == 'POST':
        url = request.form['url']
        prediction, probability = classify_url(url)
        
        session['last_url'] = url
        session['last_prediction'] = prediction
        session['last_probability'] = probability

        # Generate and save the plot
        plot_path = generate_and_save_plot()
        session['plot_path'] = plot_path

        return render_template('feedback.html', url=url, prediction=prediction, probability=probability)
    return render_template('index.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    if 'user_id' not in session:
        return redirect(url_for('signuporlogin'))
    
    url = request.form['url']
    prediction = request.form['prediction']
    feedback = request.form['feedback']
    comments = request.form['comments']
    
    # Save feedback to the database
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO feedback (url, prediction, feedback, comments) VALUES (?, ?, ?, ?)",
                   (url, prediction, feedback, comments))
    conn.commit()
    conn.close()
    
    return redirect(url_for('index'))

@app.route('/view_feedback')
def view_feedback():
    if 'user_id' not in session:
        return redirect(url_for('signuporlogin'))
    
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM feedback")
    rows = cursor.fetchall()
    conn.close()
    return render_template('view_feedback.html', feedbacks=rows)

@app.route('/delete_feedback/<int:id>', methods=['POST'])
def delete_feedback(id):
    if 'user_id' not in session:
        return redirect(url_for('signuporlogin'))
    
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM feedback WHERE id = ?", (id,))
    conn.commit()
    conn.close()
    return redirect(url_for('view_feedback'))

@app.route('/plot')
def plot():
    if 'user_id' not in session:
        return redirect(url_for('signuporlogin'))
    
    plot_path = session.get('plot_path', '')
    return render_template('plot.html', plot_path=plot_path)

@app.route('/static/plots/<filename>')
def serve_plot(filename):
    return send_from_directory(PLOT_DIR, filename)

def classify_url(url):
    # Define domain_info dictionary here or load it from a file
    domain_info = {
        'example.com': {'creation_date': datetime(2020, 1, 1), 'is_registered': True},
        'another-example.com': {'creation_date': datetime(2015, 6, 15), 'is_registered': True},
        # Add more domain info as needed
    }
    
    features = extract_features(url, domain_info)
    features_df = pd.DataFrame([features])
    features_scaled = scaler.transform(features_df)
    
    prediction = int(xgb_model.predict(features_scaled)[0])  # Convert to int
    probability = float(max(xgb_model.predict_proba(features_scaled)[0]))  # Convert to float
    
    # Map prediction to label
    type_mapping = {0: 'benign', 1: 'phishing', 2: 'defacement'}
    prediction_label = type_mapping[prediction]
    
    return prediction_label, probability

def extract_features(url, domain_info):
    from urllib.parse import urlparse
    import re
    features = {}
    
    # URL length
    features['url_length'] = len(url)
    
    # Count special characters
    features['special_chars'] = sum([1 for char in url if char in ['/', '?', '&', '=', '-', '_', '%', '.', ':']])
    
    # Count digits
    features['digit_count'] = sum([1 for char in url if char.isdigit()])
    
    # Count letters
    features['letter_count'] = sum([1 for char in url if char.isalpha()])
    
    # Check if the URL has an IP address
    features['has_ip'] = 1 if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', url) else 0
    
    # Domain-based features
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    features['domain_length'] = len(domain)
    features['subdomain_count'] = domain.count('.')
    
    # Domain days age
    domain_creation_date = domain_info.get(domain, {}).get('creation_date')
    if domain_creation_date:
        domain_days_age = (datetime.now() - domain_creation_date).days
    else:
        domain_days_age = -1  # Unknown age
    
    features['domain_days_age'] = domain_days_age
    
    # Is registered
    features['is_registered'] = 1 if domain_info.get(domain, {}).get('is_registered', False) else 0
    
    return features

def generate_and_save_plot():
    # Create the feature importance plot
    feature_names = [feature for feature in pd.DataFrame([extract_features("", {})]).columns]
    fig, ax = plt.subplots()
    ax.barh(feature_names, xgb_model.feature_importances_)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importances')

    # Save the plot to a file
    plot_filename = f'plot_{datetime.now().strftime("%Y%m%d%H%M%S")}.png'
    plot_path = os.path.join(PLOT_DIR, plot_filename)
    fig.savefig(plot_path)
    plt.close(fig)
    
    return plot_filename

if __name__ == '__main__':
    app.run(debug=True)
