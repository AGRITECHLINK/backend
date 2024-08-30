from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from PIL import Image
import numpy as np
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datetime import datetime


app = Flask(__name__)
app.config['SECRET_KEY'] = 'e232a2c5-dc92-4899-aaa0-681f7750c6ea'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agritech.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model
model = tf.keras.models.load_model('plant_disease_model_finetuned.keras')
print(f"Model loaded. Summary:")
model.summary()
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

# Load aksara_v1 model
# pipe = pipeline("text-generation", model="cropinailab/aksara_v1", use_auth_token="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", framework="tf")
# tokenizer = AutoTokenizer.from_pretrained("cropinailab/aksara_v1")
# chat_model = AutoModelForCausalLM.from_pretrained("cropinailab/aksara_v1")

print("Aksara_v1 model loaded successfully")

# Load class indices
with open('class_indices_finetuned.json', 'r') as f:
    class_indices = json.load(f)
    class_names = list(class_indices.keys())
    class_indices_inv = {v: k for k, v in class_indices.items()}

# Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    is_farmer = db.Column(db.Boolean, default=True)

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(200))
    price = db.Column(db.Float, nullable=False)
    seller_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
# class ChatMessage(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
#     content = db.Column(db.text, nullable=False)
#     timestamp = db.Column(db.DateTime, default=datetime.utcnow)
#     is_user = db.Column(db.Boolean, default=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Function to preprocess image for the model
def preprocess_image(image_path):
    image = Image.open(image_path).resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Function to analyze plant image using the model
def analyze_plant_image(image_path):
    try:
        preprocessed_image = preprocess_image(image_path)
        predictions = model.predict(preprocessed_image)
        condition_index = np.argmax(predictions[0])
        
        if condition_index >= len(class_names):
            print(f"Warning: condition_index ({condition_index}) is out of range for class_names (length: {len(class_names)})")
            return "Unknown condition"
        
        return class_names[condition_index]
    except Exception as e:
        print(f"Error in analyze_plant_image: {str(e)}")
        return "Error analyzing image"
    
# Function to get chat response
# def get_chat_response(message):
#     messages = [{"role": "user", "content": message}]
#     response = pipe(messages)
#     return response[0]['generated_text']

# Product recommendations
def recommend_products(plant_condition):
    product_recommendations = {
        "Apple___Apple_scab": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Apple___Black_rot": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Apple___Cedar_apple_rust": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Apple___healthy": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Blueberry___healthy": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Cherry_(including_sour)___healthy": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Cherry_(including_sour)___Powdery_mildew": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Corn_(maize)___Common_rust_": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Corn_(maize)___healthy": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Corn_(maize)___Northern_Leaf_Blight": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Grape___Black_rot": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Grape___Esca_(Black_Measles)": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Grape___healthy": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Orange___Haunglongbing_(Citrus_greening)": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Peach___Bacterial_spot": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Peach___healthy": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Pepper,_bell___healthy": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Potato___healthy": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Raspberry___healthy": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Soybean___healthy": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Squash___Powdery_mildew": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Strawberry___healthy": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Strawberry___Leaf_scorch": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Tomato___healthy": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Pepper__bell___Bacterial_spot": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Potato___Late_blight": ["Ridomil Gold MZ", "Revus Top", "Curzate", "Cabrio"],
        "Potato___Early_blight": ["Bravo Weather Stik", "Dithane M-45", "Quadris", "Priaxor"],
        "Tomato__Target_Spot": ["Bravo Weather Stik", "Quadris", "Fontelis"],
        "Tomato__Tomato_mosaic_virus": ["Resistant varieties", "Hygiene practices"],
        "Tomato__Tomato_YellowLeaf__Curl_Virus": ["Admire Pro", "Movento", "Resistant varieties", "Whitefly control"],
        "Tomato___Bacterial_spot": ["Kocide 3000", "Nordox 75 WG", "Serenade ASO"],
        "Tomato___Early_blight": ["Bravo Weather Stik", "Quadris", "Priaxor"],
        "Tomato___Late_blight": ["Ridomil Gold MZ", "Revus Top", "Curzate"],
        "Tomato___Leaf_Mold": ["Bravo Weather Stik", "Dithane M-45", "Serenade ASO"],
        "Tomato___Septoria_leaf_spot": ["Bravo Weather Stik", "Dithane M-45", "Quadris"],
        "Tomato___Spider_mites Two-spotted_spider_mite": ["Avid", "Oberon", "Forbid", "Kelthane"]
    }
    if plant_condition in product_recommendations:
        return product_recommendations[plant_condition]
    else:
        return ["No specific products available for this condition."]
# Routes
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        is_farmer = 'is_farmer' in request.form
        
        user = User(username=username, password=generate_password_hash(password), is_farmer=is_farmer)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful. Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)

@app.route('/analyze_plant', methods=['GET', 'POST'])
@login_required
def analyze_plant():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"Analyzing file: {filepath}")
            condition = analyze_plant_image(filepath)
            print(f"Analysis result: {condition}")
            recommendations = recommend_products(condition)
            return render_template('analysis_result.html', condition=condition, recommendations=recommendations)
    return render_template('analyze_plant.html')

@app.route('/marketplace')
@login_required
def marketplace():
    products = Product.query.all()
    return render_template('marketplace.html', products=products)

@app.route('/add_product', methods=['GET', 'POST'])
@login_required
def add_product():
    if not current_user.is_farmer:
        flash('Only farmers can add products')
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        product = Product(
            name=request.form['name'],
            description=request.form['description'],
            price=float(request.form['price']),
            seller_id=current_user.id
        )
        db.session.add(product)
        db.session.commit()
        flash('Product added successfully')
        return redirect(url_for('marketplace'))
    return render_template('add_product.html')

# @app.route('/chat', methods=['GET', 'POST'])
# @login_required
# def chat():
#     if request.method == 'POST':
#         user_message = request.form['message']
#         chat_message = ChatMessage(user_id=current_user.id, content=user_message, is_user=True)
#         db.session.add(chat_message)
        
#         # Get response from aksara_v1 model
#         ai_response = get_chat_response(user_message)
#         ai_chat_message = ChatMessage(user_id=current_user.id, content=ai_response, is_user=False)
#         db.session.add(ai_chat_message)
        
#         db.session.commit()
        
#         return jsonify({
#             'user_message': user_message,
#             'ai_response': ai_response
#         })
    
#     chat_history = ChatMessage.query.filter_by(user_id=current_user.id).order_by(ChatMessage.timestamp).all()
#     return render_template('chat.html', chat_history=chat_history)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5001)
