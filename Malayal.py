from flask import Flask, request, jsonify, send_file, redirect, url_for, render_template, session, flash
from flask_cors import CORS
import csv
import os
import time
import torch
from transformers import MarianMTModel, MarianTokenizer, TFDistilBertForSequenceClassification, DistilBertTokenizer
import tensorflow as tf
import secrets
from datetime import datetime, timedelta
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = secrets.token_hex(16)
CORS(app)

# File paths
USERS_FILE = 'users.csv'
COMMENTS_FILE = 'comments.csv'
OFFENSIVE_COMMENTS_FILE = 'offensive_comments.csv'
BLOCKED_USERS_FILE = 'blocked_users.csv'
OFFENSE_HISTORY_FILE = 'offense_history.csv'
VIDEO_PATH = 'static/video.mp4'

# Email configuration
EMAIL_SENDER = 'jerrinsiby01@gmail.com'
EMAIL_PASSWORD = 'your_app_password_here'  # Replace with your Gmail App Password
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587

# Ensure CSV files exist and have headers
for file_path, headers in [
    (USERS_FILE, ['name', 'email', 'password']),
    (COMMENTS_FILE, ['email', 'comment', 'username', 'timestamp', 'is_offensive']),
    (OFFENSIVE_COMMENTS_FILE, ['email', 'comment', 'timestamp', 'block_expiry', 'offense_count']),
    (BLOCKED_USERS_FILE, ['email', 'deactivation_date', 'reason']),
    (OFFENSE_HISTORY_FILE, ['email', 'offense_count', 'last_offense_timestamp'])
]:
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        with open(file_path, 'w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

# Load translation model and hate speech classifier
translation_model_name = "Helsinki-NLP/opus-mt-ml-en"
translator_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translator_model = MarianMTModel.from_pretrained(translation_model_name)

classification_model_path = r"D:\project\backend\hate_speech_model"
try:
    classification_model = TFDistilBertForSequenceClassification.from_pretrained(classification_model_path)
    classification_tokenizer = DistilBertTokenizer.from_pretrained(classification_model_path)
except:
    print("Using default pretrained model for classification")
    classification_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    classification_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Helper functions
def get_username(email):
    with open(USERS_FILE, 'r', encoding='utf-8-sig') as file:
        for row in csv.DictReader(file):
            if row['email'] == email:
                return row['name']
    return None

def is_user_blocked(email):
    try:
        with open(BLOCKED_USERS_FILE, 'r', encoding='utf-8-sig') as file:  # Use utf-8-sig to handle BOM
            reader = csv.DictReader(file)
            if 'email' not in reader.fieldnames:
                return False
            for row in reader:
                if row['email'] == email:
                    return True
    except FileNotFoundError:
        return False
    return False

def is_user_temp_blocked(email):
    current_time = datetime.now()
    with open(OFFENSIVE_COMMENTS_FILE, 'r', encoding='utf-8-sig') as file:
        for row in csv.DictReader(file):
            if row['email'] == email:
                block_expiry = datetime.fromisoformat(row['block_expiry'])
                if current_time < block_expiry:
                    seconds_left = int((block_expiry - current_time).total_seconds())
                    return True, seconds_left
    return False, 0

def get_offense_count(email):
    max_count = 0
    try:
        with open(OFFENSE_HISTORY_FILE, 'r', encoding='utf-8-sig') as file:
            for row in csv.DictReader(file):
                if row['email'] == email:
                    max_count = int(row['offense_count'])
                    break
    except FileNotFoundError:
        return 0
    return max_count

def send_email(to_email, username, is_warning=True):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = to_email
        if is_warning:
            msg['Subject'] = "Warning: Offensive Comment Detected"
            body = f"""
            Hello {username},
            We detected an offensive comment in your recent post. Your commenting privileges have been 
            temporarily suspended for 1 minute. This is offense {get_offense_count(to_email)} of 2.
            Please be mindful of our community guidelines. One more violation will result in permanent deactivation.
            Regards,
            The Team
            """
        else:
            msg['Subject'] = "Account Deactivated"
            body = f"""
            Hello {username},
            Your account has been permanently deactivated due to repeated violations of our community guidelines.
            This decision is final and you will not be able to register or login using this email address again.
            If you believe this is in error, please contact our support team.
            Regards,
            The Team
            """
        msg.attach(MIMEText(body, 'plain'))
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"Email sent to {to_email} - {'Warning' if is_warning else 'Deactivation'}")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

def permanently_block_user(email, reason="Repeated offensive comments"):
    with open(BLOCKED_USERS_FILE, 'a', newline='', encoding='utf-8-sig') as file:
        csv.writer(file).writerow([email, datetime.now().isoformat(), reason])
    username = get_username(email)
    if username:
        send_email(email, username, is_warning=False)
    print(f"User {email} has been permanently blocked")

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if 'user_email' not in session:
        return redirect(url_for('index'))
    if is_user_blocked(session['user_email']):
        flash("Your account has been deactivated due to community guidelines violations.")
        session.pop('user_email', None)
        session.pop('username', None)
        return redirect(url_for('index'))
    comments = []
    with open(COMMENTS_FILE, 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        comments = list(reader)
    is_blocked, seconds_left = is_user_temp_blocked(session['user_email'])
    temp_block_info = None
    if is_blocked:
        minutes_left = seconds_left // 60
        seconds_remainder = seconds_left % 60
        temp_block_info = {
            'minutes': minutes_left,
            'seconds': seconds_remainder,
            'total_seconds': seconds_left
        }
    offense_count = get_offense_count(session['user_email'])
    return render_template('dashboard.html', 
                          username=session.get('username'), 
                          comments=comments,
                          temp_block_info=temp_block_info,
                          offense_count=offense_count)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')
    if not all([name, email, password, confirm_password]):
        flash("All fields are required")
        return render_template('register.html')
    if password != confirm_password:
        flash("Passwords do not match")
        return render_template('register.html')
    if is_user_blocked(email):
        flash("This email has been blocked from registration. Please contact support.")
        return render_template('register.html')
    with open(USERS_FILE, 'r', encoding='utf-8-sig') as file:
        if any(row['email'] == email for row in csv.DictReader(file)):
            flash("User already exists")
            return render_template('register.html')
    with open(USERS_FILE, 'a', newline='', encoding='utf-8-sig') as file:
        csv.writer(file).writerow([name, email, password])
    flash("Registration successful! Please login.")
    return redirect(url_for('index'))

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')
    if is_user_blocked(email):
        flash("This account has been deactivated due to community guidelines violations.")
        return redirect(url_for('index'))
    with open(USERS_FILE, 'r', encoding='utf-8-sig') as file:
        for row in csv.DictReader(file):
            if row['email'] == email and row['password'] == password:
                session['user_email'] = email
                session['username'] = row['name']
                return redirect(url_for('dashboard'))
    flash("Invalid credentials")
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user_email', None)
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/video')
def stream_video():
    if 'user_email' not in session:
        return redirect(url_for('index'))
    return render_template('video.html')

def is_offensive_comment(comment):
    try:
        translation_input = translator_tokenizer(comment, return_tensors="pt", padding=True, truncation=True)
        translated_output = translator_model.generate(**translation_input)
        translated_text = translator_tokenizer.decode(translated_output[0], skip_special_tokens=True)
        inputs = classification_tokenizer(translated_text, return_tensors="tf", padding=True, truncation=True)
        predictions = classification_model(inputs).logits
        return bool(tf.argmax(predictions, axis=1).numpy()[0] == 1)
    except Exception as e:
        print(f"Error in offensive comment detection: {e}")
        import random
        return random.random() < 0.2

def check_expired_blocks():
    while True:
        current_time = datetime.now()
        try:
            with open(OFFENSIVE_COMMENTS_FILE, 'r', encoding='utf-8-sig') as file:
                offensive_records = list(csv.DictReader(file))
            remaining_records = []
            for record in offensive_records:
                block_expiry = datetime.fromisoformat(record['block_expiry'])
                if current_time <= block_expiry:
                    remaining_records.append(record)
            with open(OFFENSIVE_COMMENTS_FILE, 'w', newline='', encoding='utf-8-sig') as file:
                writer = csv.writer(file)
                writer.writerow(['email', 'comment', 'timestamp', 'block_expiry', 'offense_count'])
                for record in remaining_records:
                    writer.writerow([
                        record['email'], 
                        record['comment'], 
                        record['timestamp'], 
                        record['block_expiry'],
                        record['offense_count']
                    ])
        except Exception as e:
            print(f"Error checking expired blocks: {e}")
        time.sleep(1)

block_checker = threading.Thread(target=check_expired_blocks, daemon=True)
block_checker.start()

@app.route('/comment', methods=['POST'])
def comment():
    if 'user_email' not in session:
        return redirect(url_for('index'))
    email = session['user_email']
    username = session['username']
    user_comment = request.form.get('comment')
    timestamp = datetime.now().isoformat()
    if is_user_blocked(email):
        flash("Your account has been deactivated.")
        session.pop('user_email', None)
        session.pop('username', None)
        return redirect(url_for('index'))
    is_blocked, seconds_left = is_user_temp_blocked(email)
    if is_blocked:
        minutes_left = seconds_left // 60
        seconds_remainder = seconds_left % 60
        flash(f"You are temporarily blocked from commenting for {minutes_left} minutes and {seconds_remainder} seconds.")
        return redirect(url_for('dashboard'))
    if is_offensive_comment(user_comment):
        offense_count = get_offense_count(email) + 1
        block_duration = 60
        block_expiry = (datetime.now() + timedelta(seconds=block_duration)).isoformat()
        with open(OFFENSIVE_COMMENTS_FILE, 'a', newline='', encoding='utf-8-sig') as file:
            csv.writer(file).writerow([email, user_comment, timestamp, block_expiry, offense_count])
        history_records = []
        user_found = False
        try:
            with open(OFFENSE_HISTORY_FILE, 'r', encoding='utf-8-sig') as file:
                history_records = list(csv.DictReader(file))
        except FileNotFoundError:
            pass
        with open(OFFENSE_HISTORY_FILE, 'w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(['email', 'offense_count', 'last_offense_timestamp'])
            for record in history_records:
                if record['email'] == email:
                    writer.writerow([email, offense_count, timestamp])
                    user_found = True
                else:
                    writer.writerow([record['email'], record['offense_count'], record['last_offense_timestamp']])
            if not user_found:
                writer.writerow([email, offense_count, timestamp])
        if offense_count == 1:
            send_email(email, username, is_warning=True)
            flash("⚠️ Offensive comment detected! You are blocked from commenting for 1 minute. A warning email has been sent.")
            return redirect(url_for('dashboard'))
        elif offense_count == 2:
            send_email(email, username, is_warning=True)
            flash("⚠️ Second offensive comment detected! This is your final warning. One more violation will result in permanent deactivation.")
            return redirect(url_for('dashboard'))
        else:
            permanently_block_user(email)
            flash("Your account has been deactivated due to repeated offensive comments.")
            session.pop('user_email', None)
            session.pop('username', None)
            return redirect(url_for('index'))
    with open(COMMENTS_FILE, 'a', newline='', encoding='utf-8-sig') as file:
        csv.writer(file).writerow([email, user_comment, username, timestamp, "False"])
    flash("Comment added successfully")
    return redirect(url_for('dashboard'))

# API endpoints (unchanged for brevity, apply similar encoding fixes if needed)
@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    name, email, password, confirm_password = data['name'], data['email'], data['password'], data['confirm_password']
    if not all([name, email, password, confirm_password]):
        return jsonify({"error": "All fields are required"}), 400
    if password != confirm_password:
        return jsonify({"error": "Passwords do not match"}), 400
    if is_user_blocked(email):
        return jsonify({"error": "This email has been blocked from registration"}), 403
    with open(USERS_FILE, 'r', encoding='utf-8-sig') as file:
        if any(row['email'] == email for row in csv.DictReader(file)):
            return jsonify({"error": "User already exists"}), 400
    with open(USERS_FILE, 'a', newline='', encoding='utf-8-sig') as file:
        csv.writer(file).writerow([name, email, password])
    return jsonify({"message": "User registered successfully", "user": {"name": name, "email": email}}), 201

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    email, password = data['email'], data['password']
    if is_user_blocked(email):
        return jsonify({"error": "This account has been deactivated"}), 403
    with open(USERS_FILE, 'r', encoding='utf-8-sig') as file:
        for row in csv.DictReader(file):
            if row['email'] == email and row['password'] == password:
                return jsonify({"message": "Login successful"})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/api/comment', methods=['POST'])
def api_comment():
    data = request.get_json()
    email, user_comment = data['email'], data['comment']
    username = get_username(email)
    timestamp = datetime.now().isoformat()
    if is_user_blocked(email):
        return jsonify({"error": "Your account has been deactivated"}), 403
    is_blocked, seconds_left = is_user_temp_blocked(email)
    if is_blocked:
        return jsonify({"error": f"You are temporarily blocked from commenting for {seconds_left} seconds"}), 403
    if is_offensive_comment(user_comment):
        offense_count = get_offense_count(email) + 1
        block_duration = 60
        block_expiry = (datetime.now() + timedelta(seconds=block_duration)).isoformat()
        with open(OFFENSIVE_COMMENTS_FILE, 'a', newline='', encoding='utf-8-sig') as file:
            csv.writer(file).writerow([email, user_comment, timestamp, block_expiry, offense_count])
        history_records = []
        user_found = False
        try:
            with open(OFFENSE_HISTORY_FILE, 'r', encoding='utf-8-sig') as file:
                history_records = list(csv.DictReader(file))
        except FileNotFoundError:
            pass
        with open(OFFENSE_HISTORY_FILE, 'w', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(['email', 'offense_count', 'last_offense_timestamp'])
            for record in history_records:
                if record['email'] == email:
                    writer.writerow([email, offense_count, timestamp])
                    user_found = True
                else:
                    writer.writerow([record['email'], record['offense_count'], record['last_offense_timestamp']])
            if not user_found:
                writer.writerow([email, offense_count, timestamp])
        if offense_count == 1:
            send_email(email, username, is_warning=True)
            return jsonify({"warning": "Offensive comment detected! You are blocked for 1 minute. A warning email has been sent."}), 403
        elif offense_count == 2:
            send_email(email, username, is_warning=True)
            return jsonify({"warning": "Second offensive comment detected! Final warning. One more violation will result in deactivation."}), 403
        else:
            permanently_block_user(email)
            return jsonify({"error": "Your account has been deactivated due to repeated offensive comments"}), 403
    with open(COMMENTS_FILE, 'a', newline='', encoding='utf-8-sig') as file:
        csv.writer(file).writerow([email, user_comment, username, timestamp, "False"])
    return jsonify({"message": "Comment added"})

@app.route('/api/comments')
def api_get_comments():
    with open(COMMENTS_FILE, 'r', encoding='utf-8-sig') as file:
        comments = list(csv.DictReader(file))
    return jsonify(comments)

@app.route('/api/user_status')
def api_user_status():
    email = request.args.get('email')
    if not email:
        return jsonify({"error": "Email is required"}), 400
    if is_user_blocked(email):
        return jsonify({"status": "blocked", "message": "This account has been deactivated"})
    is_blocked, seconds_left = is_user_temp_blocked(email)
    if is_blocked:
        return jsonify({"status": "temp_blocked", "seconds_left": seconds_left, "message": f"Temporarily blocked for {seconds_left} seconds"})
    offense_count = get_offense_count(email)
    return jsonify({"status": "active", "offense_count": offense_count, "message": "Account is active"})

if __name__ == '__main__':
    app.run(debug=True)