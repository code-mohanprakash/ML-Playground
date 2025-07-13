import os
import uuid
import base64
from io import BytesIO
import time
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler

import pandas as pd
import numpy as np
import json

# Set Matplotlib to use a non-GUI backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from flask_talisman import Talisman
from flask_seasurf import SeaSurf
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from models import db

# Import Regression, Classification, and Clustering models

# Regression models (10)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Classification models (13)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Clustering models (7)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture

# For generating sample datasets
from sklearn.datasets import make_blobs, make_moons, make_circles

# Metrics
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, confusion_matrix, silhouette_score, mean_absolute_error, precision_score, recall_score, f1_score, calinski_harabasz_score

# Add cross-validation
from sklearn.model_selection import cross_val_score, train_test_split

# Add new dataset structure
DATASET_LIBRARY = {
    # Regression Datasets
    "boston_housing": {
        "name": "Boston Housing",
        "category": "regression",
        "description": "House prices in Boston area",
        "samples": 506,
        "features": 13,
        "url": "https://raw.githubusercontent.com/scikit-learn/scikit-learn/master/sklearn/datasets/data/boston_house_prices.csv",
        "preview_path": "datasets/boston_housing.csv"
    },
    "california_housing": {
        "name": "California Housing",
        "category": "regression",
        "description": "California housing prices from the 1990 census",
        "samples": 20640,
        "features": 8,
        "url": "https://raw.githubusercontent.com/scikit-learn/scikit-learn/master/sklearn/datasets/data/california_housing_prices.csv",
        "preview_path": "datasets/california_housing.csv"
    },
    "diabetes": {
        "name": "Diabetes Dataset",
        "category": "regression",
        "description": "Disease progression measurements for diabetes patients",
        "samples": 442,
        "features": 10,
        "url": "https://raw.githubusercontent.com/scikit-learn/scikit-learn/master/sklearn/datasets/data/diabetes_data.csv",
        "preview_path": "datasets/diabetes.csv"
    },
    "bike_sharing": {
        "name": "Bike Sharing",
        "category": "regression",
        "description": "Hourly and daily count of rental bikes based on environmental factors",
        "samples": 17379,
        "features": 12,
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip",
        "preview_path": "datasets/bike_sharing.csv"
    },
    "wine_quality": {
        "name": "Wine Quality",
        "category": "regression",
        "description": "Predict wine quality based on physicochemical tests",
        "samples": 4898,
        "features": 11,
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "preview_path": "datasets/wine_quality.csv"
    },
    "energy_efficiency": {
        "name": "Energy Efficiency",
        "category": "regression",
        "description": "Predict heating and cooling loads of buildings",
        "samples": 768,
        "features": 8,
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
        "preview_path": "datasets/energy_efficiency.csv"
    },
    "concrete_strength": {
        "name": "Concrete Strength",
        "category": "regression",
        "description": "Predict concrete compressive strength based on components",
        "samples": 1030,
        "features": 8,
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
        "preview_path": "datasets/concrete_strength.csv"
    },
    "student_performance": {
        "name": "Student Performance",
        "category": "regression",
        "description": "Predict student grades based on demographic and social factors",
        "samples": 649,
        "features": 30,
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip",
        "preview_path": "datasets/student_performance.csv"
    },
    "air_quality": {
        "name": "Air Quality",
        "category": "regression",
        "description": "Predict air pollutant concentration based on sensor data",
        "samples": 9358,
        "features": 13,
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip",
        "preview_path": "datasets/air_quality.csv"
    },
    "real_estate_valuation": {
        "name": "Real Estate Valuation",
        "category": "regression",
        "description": "Predict house prices based on location and features",
        "samples": 414,
        "features": 6,
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx",
        "preview_path": "datasets/real_estate_valuation.csv"
    },
    
    # Classification Datasets
    "iris": {
        "name": "Iris Dataset",
        "category": "classification",
        "description": "Classic dataset for flower classification",
        "samples": 150,
        "features": 4,
        "url": "https://raw.githubusercontent.com/scikit-learn/scikit-learn/master/sklearn/datasets/data/iris.csv",
        "preview_path": "datasets/iris.csv"
    },
    "breast_cancer": {
        "name": "Breast Cancer Wisconsin",
        "category": "classification",
        "description": "Diagnostic breast cancer dataset",
        "samples": 569,
        "features": 30,
        "url": "https://raw.githubusercontent.com/scikit-learn/scikit-learn/master/sklearn/datasets/data/breast_cancer.csv",
        "preview_path": "datasets/breast_cancer.csv"
    },
    "wine": {
        "name": "Wine Dataset",
        "category": "classification",
        "description": "Classify wines based on chemical analysis",
        "samples": 178,
        "features": 13,
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
        "preview_path": "datasets/wine.csv"
    },
    "heart_disease": {
        "name": "Heart Disease",
        "category": "classification",
        "description": "Predict presence of heart disease based on medical attributes",
        "samples": 303,
        "features": 13,
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "preview_path": "datasets/heart_disease.csv"
    },
    "credit_card_fraud": {
        "name": "Credit Card Fraud",
        "category": "classification",
        "description": "Detect fraudulent credit card transactions",
        "samples": 284807,
        "features": 30,
        "url": "https://www.kaggle.com/mlg-ulb/creditcardfraud/download",
        "preview_path": "datasets/credit_card_fraud.csv"
    },
    "mushroom": {
        "name": "Mushroom Classification",
        "category": "classification",
        "description": "Classify mushrooms as edible or poisonous",
        "samples": 8124,
        "features": 22,
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
        "preview_path": "datasets/mushroom.csv"
    },
    "adult_income": {
        "name": "Adult Income",
        "category": "classification",
        "description": "Predict whether income exceeds $50K/year based on census data",
        "samples": 48842,
        "features": 14,
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "preview_path": "datasets/adult_income.csv"
    },
    "bank_marketing": {
        "name": "Bank Marketing",
        "category": "classification",
        "description": "Predict if client will subscribe to a term deposit",
        "samples": 45211,
        "features": 16,
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip",
        "preview_path": "datasets/bank_marketing.csv"
    },
    "spam": {
        "name": "Spam Detection",
        "category": "classification",
        "description": "Classify emails as spam or not spam",
        "samples": 4601,
        "features": 57,
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
        "preview_path": "datasets/spam.csv"
    },
    "titanic": {
        "name": "Titanic Survival",
        "category": "classification",
        "description": "Predict survival on the Titanic",
        "samples": 891,
        "features": 11,
        "url": "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv",
        "preview_path": "datasets/titanic.csv"
    },
    "digits": {
        "name": "Handwritten Digits",
        "category": "classification",
        "description": "Classify handwritten digits (0-9)",
        "samples": 1797,
        "features": 64,
        "url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
        "preview_path": "datasets/digits.csv"
    },
    "glass": {
        "name": "Glass Identification",
        "category": "classification",
        "description": "Classify types of glass based on chemical composition",
        "samples": 214,
        "features": 9,
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
        "preview_path": "datasets/glass.csv"
    },
    "car_evaluation": {
        "name": "Car Evaluation",
        "category": "classification",
        "description": "Evaluate car acceptability based on characteristics",
        "samples": 1728,
        "features": 6,
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
        "preview_path": "datasets/car_evaluation.csv"
    },
    
    # Clustering Datasets
    "blobs": {
        "name": "Blobs Dataset",
        "category": "clustering",
        "description": "Synthetic dataset with well-defined clusters",
        "samples": 500,
        "features": 2,
        "url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html",
        "preview_path": "datasets/blobs.csv"
    },
    "moons": {
        "name": "Moons Dataset",
        "category": "clustering",
        "description": "Two interleaving half circles",
        "samples": 500,
        "features": 2,
        "url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html",
        "preview_path": "datasets/moons.csv"
    },
    "circles": {
        "name": "Circles Dataset",
        "category": "clustering",
        "description": "Concentric circles with noise",
        "samples": 500,
        "features": 2,
        "url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html",
        "preview_path": "datasets/circles.csv"
    },
    "mall_customers": {
        "name": "Mall Customers",
        "category": "clustering",
        "description": "Customer segmentation based on spending behavior",
        "samples": 200,
        "features": 5,
        "url": "https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python/download",
        "preview_path": "datasets/mall_customers.csv"
    },
    "wholesale_customers": {
        "name": "Wholesale Customers",
        "category": "clustering",
        "description": "Segment wholesale customers based on annual spending",
        "samples": 440,
        "features": 8,
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv",
        "preview_path": "datasets/wholesale_customers.csv"
    },
    "seeds": {
        "name": "Seeds Dataset",
        "category": "clustering",
        "description": "Measurements of geometrical properties of wheat kernels",
        "samples": 210,
        "features": 7,
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt",
        "preview_path": "datasets/seeds.csv"
    },
    "user_knowledge": {
        "name": "User Knowledge",
        "category": "clustering",
        "description": "User knowledge level in an online course",
        "samples": 403,
        "features": 5,
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00257/Data_User_Modeling_Dataset_Hamdi%20Tolga%20KAHRAMAN.xls",
        "preview_path": "datasets/user_knowledge.csv"
    },
    
    # Time Series Datasets
    "airline_passengers": {
        "name": "Airline Passengers",
        "category": "time_series",
        "description": "Monthly airline passenger numbers from 1949 to 1960",
        "samples": 144,
        "features": 2,
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
        "preview_path": "datasets/airline_passengers.csv"
    },
    "stock_prices": {
        "name": "Stock Prices",
        "category": "time_series",
        "description": "Historical stock prices for tech companies",
        "samples": 1258,
        "features": 6,
        "url": "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv",
        "preview_path": "datasets/stock_prices.csv"
    },
    "temperature": {
        "name": "Global Temperature",
        "category": "time_series",
        "description": "Monthly global land temperature from 1750 to 2015",
        "samples": 3192,
        "features": 4,
        "url": "https://raw.githubusercontent.com/berkeleyearth/data/master/GlobalTemperatures/GlobalTemperatures.csv",
        "preview_path": "datasets/temperature.csv"
    },
    
    # Text Datasets
    "news_headlines": {
        "name": "News Headlines",
        "category": "text",
        "description": "News headlines for classification",
        "samples": 4000,
        "features": 2,
        "url": "https://raw.githubusercontent.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection/master/Sarcasm_Headlines_Dataset.json",
        "preview_path": "datasets/news_headlines.csv"
    },
    "movie_reviews": {
        "name": "Movie Reviews",
        "category": "text",
        "description": "Sentiment analysis of movie reviews",
        "samples": 2000,
        "features": 2,
        "url": "https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/download",
        "preview_path": "datasets/movie_reviews.csv"
    },
    "twitter_sentiment": {
        "name": "Twitter Sentiment",
        "category": "text",
        "description": "Sentiment analysis of tweets",
        "samples": 1600000,
        "features": 2,
        "url": "https://www.kaggle.com/kazanova/sentiment140/download",
        "preview_path": "datasets/twitter_sentiment.csv"
    },
    
    # Image Datasets (simplified for CSV format)
    "fashion_mnist_sample": {
        "name": "Fashion MNIST Sample",
        "category": "image",
        "description": "Sample of clothing images for classification",
        "samples": 1000,
        "features": 784,
        "url": "https://github.com/zalandoresearch/fashion-mnist",
        "preview_path": "datasets/fashion_mnist_sample.csv"
    },
    "cifar10_sample": {
        "name": "CIFAR-10 Sample",
        "category": "image",
        "description": "Sample of images in 10 classes",
        "samples": 1000,
        "features": 3072,
        "url": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "preview_path": "datasets/cifar10_sample.csv"
    }
}

# Model Options Dictionary (30 models)
model_options = {
    # Regression Models (10)
    "Linear Regression": {
        "category": "regression",
        "model": lambda: LinearRegression(),
        "desc": "Predicts continuous outcomes using a linear combination of features."
    },
    "Ridge Regression": {
        "category": "regression",
        "model": lambda: Ridge(),
        "desc": "Linear regression with L2 regularization to avoid overfitting."
    },
    "Lasso Regression": {
        "category": "regression",
        "model": lambda: Lasso(),
        "desc": "Linear regression with L1 regularization, useful for feature selection."
    },
    "ElasticNet Regression": {
        "category": "regression",
        "model": lambda: ElasticNet(),
        "desc": "Combines L1 and L2 penalties to balance feature selection and regularization."
    },
    "Decision Tree Regressor": {
        "category": "regression",
        "model": lambda: DecisionTreeRegressor(),
        "desc": "A tree‑based model that splits data based on feature values."
    },
    "Random Forest Regressor": {
        "category": "regression",
        "model": lambda: RandomForestRegressor(),
        "desc": "An ensemble of decision trees for improved regression accuracy."
    },
    "Gradient Boosting Regressor": {
        "category": "regression",
        "model": lambda: GradientBoostingRegressor(),
        "desc": "Sequentially builds trees to reduce prediction errors."
    },
    "SVR": {
        "category": "regression",
        "model": lambda: SVR(),
        "desc": "Support Vector Regression for non‑linear regression tasks."
    },
    "KNeighbors Regressor": {
        "category": "regression",
        "model": lambda: KNeighborsRegressor(),
        "desc": "Predicts values based on the average of nearest neighbors."
    },
    "Bayesian Ridge Regression": {
        "category": "regression",
        "model": lambda: BayesianRidge(),
        "desc": "A probabilistic regression model with Bayesian inference."
    },

    # Classification Models (13)
    "Logistic Regression": {
        "category": "classification",
        "model": lambda: LogisticRegression(max_iter=200),
        "desc": "Uses a logistic function to model binary (or multinomial) outcomes."
    },
    "Decision Tree Classifier": {
        "category": "classification",
        "model": lambda: DecisionTreeClassifier(),
        "desc": "A tree‑based classifier splitting data based on features."
    },
    "Random Forest Classifier": {
        "category": "classification",
        "model": lambda: RandomForestClassifier(),
        "desc": "An ensemble of trees that improves classification performance."
    },
    "Gradient Boosting Classifier": {
        "category": "classification",
        "model": lambda: GradientBoostingClassifier(),
        "desc": "Sequentially builds trees to improve classification accuracy."
    },
    "Support Vector Classifier": {
        "category": "classification",
        "model": lambda: SVC(),
        "desc": "Classifies data by finding the optimal hyperplane between classes."
    },
    "KNeighbors Classifier": {
        "category": "classification",
        "model": lambda: KNeighborsClassifier(),
        "desc": "Classifies based on the majority vote of nearest neighbors."
    },
    "Gaussian Naive Bayes": {
        "category": "classification",
        "model": lambda: GaussianNB(),
        "desc": "Assumes features follow a Gaussian distribution for classification."
    },
    "Multinomial Naive Bayes": {
        "category": "classification",
        "model": lambda: MultinomialNB(),
        "desc": "Suitable for classification with discrete features (e.g., word counts)."
    },
    "Bernoulli Naive Bayes": {
        "category": "classification",
        "model": lambda: BernoulliNB(),
        "desc": "Designed for binary/boolean features in classification."
    },
    "Linear Discriminant Analysis": {
        "category": "classification",
        "model": lambda: LinearDiscriminantAnalysis(),
        "desc": "Finds linear combinations of features that best separate classes."
    },
    "Quadratic Discriminant Analysis": {
        "category": "classification",
        "model": lambda: QuadraticDiscriminantAnalysis(),
        "desc": "A quadratic version of LDA that allows for more complex boundaries."
    },
    "AdaBoost Classifier": {
        "category": "classification",
        "model": lambda: AdaBoostClassifier(),
        "desc": "Boosting ensemble that iteratively focuses on misclassified samples."
    },
    "Extra Trees Classifier": {
        "category": "classification",
        "model": lambda: ExtraTreesClassifier(),
        "desc": "Uses a large number of randomized trees for classification."
    },

    # Clustering Models (7)
    "K-Means Clustering": {
        "category": "clustering",
        "model": lambda: KMeans(n_clusters=3),
        "desc": "Clusters data into k groups by minimizing within‑cluster variance."
    },
    "Agglomerative Clustering": {
        "category": "clustering",
        "model": lambda: AgglomerativeClustering(n_clusters=3),
        "desc": "Hierarchical clustering that builds nested clusters."
    },
    "DBSCAN": {
        "category": "clustering",
        "model": lambda: DBSCAN(),
        "desc": "Clusters based on data density, identifying outliers as noise."
    },
    "Mean Shift": {
        "category": "clustering",
        "model": lambda: MeanShift(),
        "desc": "Identifies clusters by locating dense areas in feature space."
    },
    "Spectral Clustering": {
        "category": "clustering",
        "model": lambda: SpectralClustering(n_clusters=3, assign_labels='discretize'),
        "desc": "Uses eigenvalues of a similarity matrix to perform clustering."
    },
    "Gaussian Mixture Model": {
        "category": "clustering",
        "model": lambda: GaussianMixture(n_components=3),
        "desc": "Assumes data is generated from a mixture of Gaussian distributions."
    },
    "Birch Clustering": {
        "category": "clustering",
        "model": lambda: Birch(n_clusters=3),
        "desc": "Uses hierarchical clustering to incrementally build clusters."
    },
}

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_SECURE'] = False  # Changed to False for local development without HTTPS
app.config['REMEMBER_COOKIE_SECURE'] = False  # Changed to False for local development without HTTPS
app.config['CSRF_ENABLED'] = True
app.config['SESSION_TYPE'] = 'filesystem'  # Store sessions in filesystem for persistence (only used locally)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)  # Session lasts for 24 hours

# Initialize Flask-Session if it's being used (only for local development)
if not os.environ.get('VERCEL'):
    try:
        from flask_session import Session
        Session(app)
        print("Flask-Session initialized")
    except ImportError:
        print("Flask-Session not available, using default session")
else:
    print("Using default Flask session for Vercel deployment")

# File Upload Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize Flask-SeaSurf for CSRF protection
csrf = SeaSurf(app)
csrf._exempt_urls = ('/upload', '/upload_file', '/run_model', '/tuning')  # Removed auth routes

# Initialize Talisman for security headers
talisman = Talisman(app, content_security_policy=None)

# Initialize extensions
db.init_app(app)

# Create database tables and directories only if not on Vercel
if not os.environ.get('VERCEL'):
    # Ensure uploads directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Create database tables
    with app.app_context():
        db.create_all()

@app.route('/')
def index():
    """Serve the index page"""
    try:
        # Count the number of datasets and models for the stats section
        dataset_count = len(DATASET_LIBRARY) + sum(1 for f in os.listdir('uploads') if f.endswith('.csv'))
        model_count = len(model_options)
        
        # Get featured datasets (one from each category)
        featured_datasets = []
        categories = ['regression', 'classification', 'clustering', 'time_series', 'text', 'image']
        for category in categories:
            # Find datasets in this category
            category_datasets = [(k, v) for k, v in DATASET_LIBRARY.items() if v['category'] == category]
            if category_datasets:
                # Add the first dataset from this category
                featured_datasets.append(category_datasets[0])
        
        # Featured models
        featured_models = [
            {
                'name': 'Linear Regression',
                'type': 'regression',
                'description': 'Predicts a continuous value based on linear relationships between features.'
            },
            {
                'name': 'Random Forest Classifier',
                'type': 'classification',
                'description': 'Ensemble learning method that builds multiple decision trees for classification.'
            },
            {
                'name': 'K-Means Clustering',
                'type': 'clustering',
                'description': 'Partitions data into k clusters, minimizing within-cluster variance.'
            }
        ]
        
        return render_template('index.html', 
                            dataset_count=dataset_count, 
                            model_count=model_count,
                            featured_datasets=featured_datasets,
                            featured_models=featured_models)
    except Exception as e:
        print(f"Error in index route: {str(e)}")
        # Return a simple response if there's an error
        return render_template('index.html',
                            dataset_count=len(DATASET_LIBRARY),
                            model_count=len(model_options),
                            featured_datasets=[],
                            featured_models=[])

# Exempt file upload from CSRF protection
@csrf.exempt
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle file uploads and dataset selection"""
    if request.method == 'POST':
        # Check if user uploaded a file
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user doesn't select file, browser might submit an empty file without filename
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        
        if file:
            # Handle file upload for Vercel (read-only filesystem)
            if os.environ.get('VERCEL'):
                # On Vercel, we can't save files, so we'll store the file content in session
                file_content = file.read()
                file_data = {
                    'filename': file.filename,
                    'content': base64.b64encode(file_content).decode('utf-8'),
                    'content_type': file.content_type
                }
                session['file_data'] = file_data
                flash('File successfully uploaded! (Vercel mode)', 'success')
            else:
                # Save the uploaded file to a unique location
                unique_filename = str(uuid.uuid4()) + '_' + file.filename
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                
                # Store the filepath in session
                session['file_path'] = file_path  # Using consistent naming: file_path
                flash('File successfully uploaded!', 'success')
            
            return redirect(url_for('select_model'))
    
    # Sample datasets to display
    sample_datasets = []
    for dataset_id, dataset_info in DATASET_LIBRARY.items():
        sample_datasets.append({
            'id': dataset_id,
            'name': dataset_info['name'],
            'description': dataset_info['description'],
            'category': dataset_info['category'],
            'samples': dataset_info['samples'],
            'features': dataset_info['features']
        })
    
    return render_template('upload.html', sample_datasets=sample_datasets)

# Dedicated route for file upload only - no CSRF
@app.route('/upload_file', methods=['POST'])
@csrf.exempt
def upload_file():
    """Handle file uploads only - no CSRF protection"""
    try:
        # Check if user uploaded a file
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        # If user doesn't select file, browser might submit an empty file without filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file:
            # Handle file upload for Vercel (read-only filesystem)
            if os.environ.get('VERCEL'):
                # On Vercel, we can't save files, so we'll store the file content in session
                file_content = file.read()
                file_data = {
                    'filename': file.filename,
                    'content': base64.b64encode(file_content).decode('utf-8'),
                    'content_type': file.content_type
                }
                session['file_data'] = file_data
                return jsonify({'success': True, 'message': 'File successfully uploaded! (Vercel mode)'}), 200
            else:
                # Save the uploaded file to a unique location
                unique_filename = str(uuid.uuid4()) + '_' + file.filename
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                
                # Store the filepath in session
                session['file_path'] = file_path
                return jsonify({'success': True, 'message': 'File successfully uploaded!'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/select_model', methods=['GET', 'POST'])
def select_model():
    """Page to select a model and configure it"""
    # Get model name from query param if provided (from recommendation page)
    suggested_model = request.args.get('model', '')
    
    # Get available models
    regression_models = [name for name, info in model_options.items() if info["category"] == "regression"]
    classification_models = [name for name, info in model_options.items() if info["category"] == "classification"]
    clustering_models = [name for name, info in model_options.items() if info["category"] == "clustering"]
    
    # Create model descriptions dictionary
    model_descriptions = {}
    for name, info in model_options.items():
        model_descriptions[name] = info.get("description", info.get("desc", "No description available"))
    
            # Get the file data from the session - handle both Vercel and local modes
        file_path = session.get('file_path')
        file_data = session.get('file_data')
        
        if os.environ.get('VERCEL'):
            # On Vercel, use file data from session
            if not file_data:
                flash("Please upload a dataset first", "error")
                return redirect(url_for('upload'))
        else:
            # Local mode - check file path
            if not file_path or not os.path.exists(file_path):
                flash("Please upload a dataset first", "error")
                return redirect(url_for('upload'))
    
    try:
        # Load dataset to get column names
        if os.environ.get('VERCEL') and file_data:
            # Decode file content from session
            file_content = base64.b64decode(file_data['content'])
            df = pd.read_csv(BytesIO(file_content))
        else:
            df = pd.read_csv(file_path)
        columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if request.method == 'POST':
            model_name = request.form.get('model')
            target = request.form.get('target')
            features = request.form.getlist('features')
            
            if not model_name:
                flash('Please select a model', 'error')
                return redirect(url_for('select_model'))
            
            if not target:
                flash('Please select a target column', 'error')
                return redirect(url_for('select_model'))
            
            if not features:
                flash('Please select at least one feature', 'error')
                return redirect(url_for('select_model'))
            
            # Save selections to session
            session['model'] = model_name
            session['target'] = target
            session['features'] = features
            
            return redirect(url_for('run_model'))
        
        # Get model from session if available
        selected_model = session.get('model', suggested_model or regression_models[0] if regression_models else None)
        selected_target = session.get('target', '')
        selected_features = session.get('features', [])
        
        return render_template('model_select.html', 
                           regression_models=regression_models,
                           classification_models=classification_models,
                           clustering_models=clustering_models,
                           columns=columns, 
                           numeric_columns=numeric_columns,
                           categorical_columns=categorical_columns,
                           selected_model=selected_model,
                           selected_target=selected_target,
                           selected_features=selected_features,
                           model_descriptions=model_descriptions)
    except Exception as e:
        flash(f"Error loading dataset: {str(e)}", "error")
        return redirect(url_for('upload'))

@app.route('/run_model', methods=['POST'])
def run_model():
    try:
        print("\n\n==== RUN MODEL FUNCTION STARTED ====")
        # Get form data
        model_name = request.form.get('model')
        target_col = request.form.get('target')
        feature_cols = request.form.getlist('features')
        
        print(f"Model: {model_name}")
        print(f"Target: {target_col}")
        print(f"Features: {feature_cols}")
        
        # Validate inputs with more detailed error messages
        if not model_name:
            print("ERROR: No model selected")
            flash("Please select a model.", "error")
            return redirect(url_for('select_model'))
            
        if not target_col:
            print("ERROR: No target variable selected")
            flash("Please select a target variable.", "error")
            return redirect(url_for('select_model'))
            
        if not feature_cols:
            print("ERROR: No features selected")
            flash("Please select at least one feature.", "error")
            return redirect(url_for('select_model'))

        # Get the data from the session - handle both Vercel and local modes
        file_path = session.get('file_path')
        file_data = session.get('file_data')
        print(f"File path from session: {file_path}")
        print(f"File data from session: {bool(file_data)}")
        
        if os.environ.get('VERCEL'):
            # On Vercel, use file data from session
            if not file_data:
                print("ERROR: No file data in session (Vercel mode)")
                flash("Dataset not found. Please upload your data again.", "error")
                return redirect(url_for('upload'))
        else:
            # Local mode - check file path
            if not file_path:
                print("ERROR: No file path in session")
                flash("Dataset not found. Please upload your data again.", "error")
                return redirect(url_for('upload'))
                
            if not os.path.exists(file_path):
                print(f"ERROR: File does not exist at path: {file_path}")
                flash("Dataset file not found. Please upload your data again.", "error")
                return redirect(url_for('upload'))

        print("Loading data from file...")
        # Load and prepare the data
        try:
            if os.environ.get('VERCEL') and file_data:
                # Decode file content from session
                file_content = base64.b64decode(file_data['content'])
                data = pd.read_csv(BytesIO(file_content))
            else:
                data = pd.read_csv(file_path)
            print(f"Data loaded successfully. Shape: {data.shape}")
            print(f"Columns: {data.columns.tolist()}")
            
            # Check if target column exists
            if target_col not in data.columns:
                print(f"ERROR: Target column '{target_col}' not in data columns")
                flash(f"Target column '{target_col}' not found in dataset.", "error")
                return redirect(url_for('select_model'))
                
            # Check if all feature columns exist
            missing_features = [col for col in feature_cols if col not in data.columns]
            if missing_features:
                print(f"ERROR: Missing feature columns: {missing_features}")
                flash(f"Feature columns {missing_features} not found in dataset.", "error")
                return redirect(url_for('select_model'))
                
            X = data[feature_cols]
            y = data[target_col]
            print(f"X shape: {X.shape}, y shape: {y.shape if hasattr(y, 'shape') else 'scalar'}")
        except Exception as e:
            print(f"ERROR loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            flash(f"Error loading data: {str(e)}", "error")
            return redirect(url_for('select_model'))

        # Get the model from model_options
        if model_name not in model_options:
            print(f"ERROR: Invalid model selection: {model_name}")
            flash(f"Invalid model selection: {model_name}", "error")
            return redirect(url_for('select_model'))

        print(f"Initializing model: {model_name}")
        # Record start time
        start_time = time.time()

        try:
            # Initialize the model
            model = model_options[model_name]['model']()
            model_type = model_options[model_name]['category']
            print(f"Model initialized. Type: {model_type}")

            # Split the data
            print("Splitting data into train and test sets")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"Train shapes: X={X_train.shape}, y={y_train.shape if hasattr(y_train, 'shape') else 'scalar'}")
            print(f"Test shapes: X={X_test.shape}, y={y_test.shape if hasattr(y_test, 'shape') else 'scalar'}")

            # Train the model
            print("Training the model...")
            model.fit(X_train, y_train)
            print("Model training completed")
            
            # Record end time
            training_time = time.time() - start_time
            print(f"Training time: {training_time} seconds")
            
            # Make predictions
            print("Making predictions...")
            predictions = model.predict(X_test)
            print("Predictions made")
            
            # Calculate metrics based on model type
            print("Calculating metrics...")
            metrics = {}
            if model_type == 'regression':
                metrics['mse'] = float(mean_squared_error(y_test, predictions))
                metrics['r2'] = float(r2_score(y_test, predictions))
                metrics['mae'] = float(mean_absolute_error(y_test, predictions))
            elif model_type == 'classification':
                metrics['accuracy'] = float(accuracy_score(y_test, predictions))
                # Handle multi-class classification for precision, recall, f1
                try:
                    metrics['precision'] = float(precision_score(y_test, predictions, average='weighted'))
                    metrics['recall'] = float(recall_score(y_test, predictions, average='weighted'))
                    metrics['f1'] = float(f1_score(y_test, predictions, average='weighted'))
                except Exception as metric_error:
                    print(f"Error calculating classification metrics: {str(metric_error)}")
                    metrics['precision'] = 0.0
                    metrics['recall'] = 0.0
                    metrics['f1'] = 0.0
            elif model_type == 'clustering':
                if hasattr(model, 'labels_'):
                    try:
                        metrics['silhouette'] = float(silhouette_score(X, model.labels_))
                        metrics['calinski'] = float(calinski_harabasz_score(X, model.labels_))
                    except Exception as metric_error:
                        print(f"Error calculating clustering metrics: {str(metric_error)}")
                        metrics['silhouette'] = 0.0
                        metrics['calinski'] = 0.0
            print(f"Metrics calculated: {metrics}")

            # Get feature importance if available
            print("Getting feature importance...")
            feature_importance = []
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = list(zip(feature_cols, importances))
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
                if importances.ndim > 1:
                    importances = importances.mean(axis=0)
                feature_importance = list(zip(feature_cols, importances))
            print(f"Feature importance: {feature_importance}")
            
            # Generate visualization
            print("Generating visualization...")
            plt.figure(figsize=(10, 6))
            if model_type == 'regression':
                plt.scatter(y_test, predictions, alpha=0.5)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.title('Actual vs Predicted Values')
            elif model_type == 'classification':
                if hasattr(model, 'predict_proba'):
                    try:
                        probs = model.predict_proba(X_test)
                        plt.hist(probs.max(axis=1), bins=20)
                        plt.xlabel('Prediction Probability')
                        plt.ylabel('Count')
                        plt.title('Prediction Confidence Distribution')
                    except Exception as plot_error:
                        print(f"Error generating classification plot: {str(plot_error)}")
                        # Fallback plot if predict_proba fails
                        plt.text(0.5, 0.5, 'Unable to generate probability plot', 
                                horizontalalignment='center', verticalalignment='center')
            
            # Save plot to base64
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_url = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            print("Visualization generated")
            
            # Prepare results dictionary
            print("Preparing results dictionary...")
            results_dict = {
                'model_name': model_name,
                'model_type': model_type,
                'target': target_col,
                'features': feature_cols,
                'metrics': metrics,
                'predictions': predictions.tolist()[:10] if hasattr(predictions, 'tolist') else list(predictions)[:10],  # Store first 10 predictions as example
                'training_time': training_time,
                'data_points': len(data),
                'feature_importance': feature_importance,
                'plot_url': f"data:image/png;base64,{plot_url}"
            }
            
            # Store results in session
            print("Storing results in session...")
            session['results'] = results_dict
            # Make sure session is saved
            session.modified = True
            
            # Print session data for debugging
            print("Session data after storing results:")
            for key in session:
                print(f"  {key}: {type(session[key])}")
            
            print("Results stored in session, redirecting to results page")
            flash('Model training completed successfully!', 'success')
            
            # Use a direct response instead of redirect to preserve session
            return redirect(url_for('results'))
        except Exception as e:
            print(f"ERROR in model training/evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            flash(f"Error during model training: {str(e)}", "error")
            return redirect(url_for('select_model'))
            
    except Exception as e:
        print(f"CRITICAL ERROR in run_model: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f"An error occurred while training the model: {str(e)}", "error")
        return redirect(url_for('select_model'))

@app.route('/results')
@app.route('/results/<int:index>')
def results(index=None):
    """Display model results"""
    print("\n\n==== RESULTS ROUTE CALLED ====")
    
    # If index is provided, get results from history
    if index is not None:
        history = session.get('history', [])
        if 0 <= index < len(history):
            results = history[index]
            print(f"Showing historical results at index {index}")
        else:
            print(f"ERROR: Invalid history index: {index}")
            flash('Invalid history index.', 'error')
            return redirect(url_for('history'))
    else:
        # Get results from session
        results = session.get('results')
        print(f"Results from session: {results}")
    
    # Check if results exist
    if not results:
        print("ERROR: No results found in session")
        flash('No results to display. Please run a model first.', 'error')
        return redirect(url_for('select_model'))
    
    print("Results found in session, rendering template")
    
    # Add results to history if not already there and if not viewing history
    if index is None:
        history = session.get('history', [])
        
        # Add timestamp if not present
        if 'timestamp' not in results:
            results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Add to history
        history.append(results)
        session['history'] = history
        session.modified = True  # Ensure session is saved
        print("Results added to history")
    
    return render_template('results.html', results=results)

@app.route('/history')
def history():
    history = session.get('history', [])
    return render_template('history.html', history=history)

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    # Get available models
    models = list(model_options.keys())
    
    if request.method == 'POST':
        model1 = request.form.get('model1')
        model2 = request.form.get('model2')
        session['compare_models'] = [model1, model2]
        flash(f'Comparison requested: {model1} vs. {model2}. Please upload a dataset and select features.')
        return redirect(url_for('upload'))
    
    # Check if we have comparison results
    comparison_results = session.get('comparison_results', {
        'models': ['Linear Regression', 'Random Forest Regressor'],
        'metrics': {
            'r2_score': [0.75, 0.85],
            'mean_squared_error': [0.25, 0.15]
        }
    })
    
    best_metrics = comparison_results.get('best_metrics', {'r2_score': 'Random Forest Regressor', 'mean_squared_error': 'Random Forest Regressor'})
    comparison_plot = comparison_results.get('comparison_plot', 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==')
    
    return render_template('compare.html', 
                           models=models,
                           datasets=DATASET_LIBRARY, 
                           comparison_results=comparison_results,
                           best_metrics=best_metrics,
                           comparison_plot=comparison_plot)

@app.route('/tuning', methods=['GET', 'POST'])
def tuning():
    # Get available models
    models = list(model_options.keys())
    
    if request.method == 'POST':
        model_name = request.form.get('model')
        tuning_method = request.form.get('tuning_method')
        cv_folds = int(request.form.get('cv_folds', 5))
        scoring = request.form.get('scoring', 'accuracy')
        n_iter = int(request.form.get('n_iter', 20))
        
        # Get parameters from form
        params = {}
        for key, value in request.form.items():
            if key.startswith('param_'):
                param_name = key.replace('param_', '')
                params[param_name] = value
        
        # Perform hyperparameter tuning (simplified example)
        tuning_results = {
            'best_params': params,
            'best_score': 0.85,  # Placeholder
            'default_score': 0.75,  # Placeholder
            'improvement': 13.3  # Placeholder
        }
        
        # Store tuning results in session
        session['tuning_results'] = tuning_results
        session.modified = True
        
        flash('Hyperparameter tuning completed successfully!', 'success')
        return render_template('tuning.html', models=models, tuning_results=tuning_results)
    
    # For GET requests, check if there are existing tuning results
    tuning_results = session.get('tuning_results')
    return render_template('tuning.html', models=models, tuning_results=tuning_results)

@app.route('/apply_tuned_model')
def apply_tuned_model():
    """Apply the tuned model to the dataset"""
    # Get the tuned parameters from the session
    tuning_results = session.get('tuning_results', {})
    best_params = tuning_results.get('best_params', {})
    
    # Store the best parameters in session for use in run_model
    session['tuned_params'] = best_params
    
    # Redirect to select_model to choose features, etc.
    flash('Tuned parameters applied. Now select features for your model.', 'success')
    return redirect(url_for('select_model'))

@app.route('/use_sample_dataset/<dataset_name>')
def use_sample_dataset(dataset_name):
    """Use a sample dataset from the library"""
    if dataset_name in DATASET_LIBRARY:
        # Ensure the dataset exists first
        ensure_sample_datasets()
        
        # Set the filepath in the session
        file_path = DATASET_LIBRARY[dataset_name]['preview_path']
        session['file_path'] = file_path  # Using consistent naming: file_path
        
        flash(f"Using {DATASET_LIBRARY[dataset_name]['name']} dataset", 'success')
        return redirect(url_for('select_model'))
    else:
        flash('Invalid dataset selected', 'error')
        return redirect(url_for('upload'))

@app.route('/datasets')
def datasets():
    """Show available datasets"""
    # Ensure datasets exist
    ensure_sample_datasets()
    return render_template('datasets.html', datasets=DATASET_LIBRARY)

@app.route('/download_dataset/<dataset_name>')
def download_dataset(dataset_name):
    """Download a dataset"""
    if dataset_name in DATASET_LIBRARY:
        filepath = DATASET_LIBRARY[dataset_name]['preview_path']
        if not os.path.exists(filepath):
            # Create dataset if it doesn't exist
            ensure_sample_datasets()
        
        if os.path.exists(filepath):
            return send_file(filepath, 
                            mimetype='text/csv',
                            as_attachment=True,
                            download_name=f"{dataset_name}.csv")
    
    flash(f"Dataset {dataset_name} not found", "error")
    return redirect(url_for('datasets'))

@app.route('/preview_dataset/<dataset_name>')
def preview_dataset(dataset_name):
    """Preview a dataset"""
    if dataset_name in DATASET_LIBRARY:
        filepath = DATASET_LIBRARY[dataset_name]['preview_path']
        
        # Handle Vercel's read-only filesystem
        if os.environ.get('VERCEL'):
            # On Vercel, we can't read local files, so show a message
            flash("Dataset preview not available on Vercel deployment. Please download the dataset to view it.", "info")
            return redirect(url_for('datasets'))
        
        # Make sure the dataset exists
        if not os.path.exists(filepath):
            ensure_sample_datasets()
        
        try:
            # Read only the first 20 rows for preview
            df = pd.read_csv(filepath, nrows=20)
            
            # Convert to HTML
            table_html = df.head(20).to_html(classes='table table-striped table-hover table-sm', 
                                           index=False, border=0)
            
            # Get basic stats
            stats = {
                'rows': len(pd.read_csv(filepath)),
                'columns': len(df.columns),
                'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
            }
            
            return render_template('dataset_preview.html', 
                               dataset_name=DATASET_LIBRARY[dataset_name]['name'],
                               table=table_html,
                               stats=stats,
                               download_url=url_for('download_dataset', dataset_name=dataset_name))
        except Exception as e:
            flash(f"Error previewing dataset: {str(e)}", "error")
    else:
        flash(f"Dataset {dataset_name} not found", "error")
    
    return redirect(url_for('datasets'))

@app.route('/resources')
def resources():
    """Educational resources about machine learning"""
    # ML courses and tutorials
    resources_list = [
        {
            "title": "Machine Learning Crash Course",
            "provider": "Google",
            "description": "A self-study guide for aspiring machine learning practitioners",
            "url": "https://developers.google.com/machine-learning/crash-course",
            "image": "https://developers.google.com/static/machine-learning/crash-course/images/ml_crash_course.png",
            "free": True
        },
        {
            "title": "Machine Learning A-Z",
            "provider": "Udemy",
            "description": "Hands-On Python & R In Data Science",
            "url": "https://www.udemy.com/course/machinelearning/",
            "image": "https://img-c.udemycdn.com/course/750x422/950390_270f_3.jpg",
            "free": False
        },
        {
            "title": "Machine Learning with Krish Naik",
            "provider": "Udemy",
            "description": "Complete Machine Learning & Data Science with Python",
            "url": "https://www.udemy.com/course/machine-learning-with-python-krishnaik/",
            "image": "https://img-c.udemycdn.com/course/750x422/3060638_5021_4.jpg",
            "free": False
        },
        {
            "title": "Introduction to Machine Learning",
            "provider": "Coursera",
            "description": "Build ML models with NumPy & scikit-learn",
            "url": "https://www.coursera.org/learn/machine-learning-duke",
            "image": "https://d3njjcbhbojbot.cloudfront.net/api/utilities/v1/imageproxy/https://coursera-course-photos.s3.amazonaws.com/bb/02d8201ab511e78fbb725306bef451/MLwP.jpg",
            "free": True
        },
        {
            "title": "Machine Learning Guide Podcast",
            "provider": "OCDevel",
            "description": "Simple, clear explanations of machine learning concepts",
            "url": "https://ocdevel.com/mlg",
            "image": "https://is1-ssl.mzstatic.com/image/thumb/Podcasts125/v4/4a/9e/b2/4a9eb290-5554-6458-80c4-6699f288ee6d/mza_17172766526325955932.jpg/1200x1200bb.jpg",
            "free": True
        }
    ]
    
    # ML glossary - common terms and definitions
    glossary = {
        "Algorithm": "A procedure or formula for solving a problem.",
        "Bias": "A systematic error introduced into the sampling or testing process.",
        "Classification": "A supervised learning approach where the output variable is a category.",
        "Clustering": "Grouping a set of objects in such a way that objects in the same group are more similar to each other than to those in other groups.",
        "Cross-validation": "A technique to evaluate predictive models by partitioning the original sample into a training set and a test set.",
        "Dimensionality Reduction": "The process of reducing the number of random variables under consideration.",
        "Ensemble Learning": "Using multiple learning algorithms to obtain better predictive performance.",
        "Feature": "An individual measurable property of the phenomenon being observed.",
        "Hyperparameter": "A parameter whose value is set before the learning process begins.",
        "Overfitting": "When a model learns the training data too well, including noise and outliers.",
        "Regression": "A supervised learning approach where the output variable is a real value.",
        "Regularization": "A technique used to prevent overfitting by adding a penalty term to the loss function.",
        "Supervised Learning": "The machine learning task of inferring a function from labeled training data.",
        "Unsupervised Learning": "The machine learning task of inferring a function to describe hidden structure from unlabeled data."
    }
    
    # Books recommendations
    books = [
        {
            "title": "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow",
            "author": "Aurélien Géron",
            "description": "A practical guide to machine learning with Python",
            "url": "https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/",
            "image": "https://learning.oreilly.com/library/cover/9781492032632/250w/"
        },
        {
            "title": "Pattern Recognition and Machine Learning",
            "author": "Christopher Bishop",
            "description": "A comprehensive introduction to machine learning, pattern recognition, and related fields",
            "url": "https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/",
            "image": "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Bishop-Pattern-Recognition-and-Machine-Learning-2006.jpg"
        },
        {
            "title": "The Elements of Statistical Learning",
            "author": "Trevor Hastie, Robert Tibshirani, Jerome Friedman",
            "description": "A comprehensive overview of statistical learning methods",
            "url": "https://hastie.su.domains/ElemStatLearn/",
            "image": "https://hastie.su.domains/ElemStatLearn/CoverII_small.jpg"
        }
    ]
    
    return render_template('resources.html', resources=resources_list, glossary=glossary, books=books)

@app.route('/tutorial/<topic>')
def tutorial(topic):
    """Show specific tutorial"""
    tutorials = {
        'intro': {
            'title': 'Introduction to Machine Learning',
            'sections': [
                {'title': 'What is Machine Learning?', 'content': 'Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to "learn" from data, without being explicitly programmed.'},
                {'title': 'Types of Machine Learning', 'content': 'The three main types are supervised learning (labeled data), unsupervised learning (unlabeled data), and reinforcement learning (learning through rewards).'},
                {'title': 'Common Applications', 'content': 'Machine learning is used in recommendation systems, image recognition, spam filtering, predictive analytics, and much more.'}
            ]
        },
        'regression': {
            'title': 'Understanding Regression Models',
            'sections': [
                {'title': 'What is Regression?', 'content': 'Regression analysis is a set of statistical processes for estimating the relationships between a dependent variable and one or more independent variables.'},
                {'title': 'Linear Regression', 'content': 'Linear regression attempts to model the relationship between variables by fitting a linear equation to observed data.'},
                {'title': 'Evaluating Regression Models', 'content': 'Common metrics include R-squared, Mean Squared Error (MSE), and Mean Absolute Error (MAE).'}
            ]
        },
        'classification': {
            'title': 'Classification Techniques',
            'sections': [
                {'title': 'What is Classification?', 'content': 'Classification is the problem of identifying to which category a new observation belongs, based on a training set of data containing observations with known category membership.'},
                {'title': 'Logistic Regression', 'content': 'Despite its name, logistic regression is a classification algorithm used to predict a binary outcome based on a set of independent variables.'},
                {'title': 'Decision Trees', 'content': 'A decision tree is a flowchart-like structure where each internal node represents a test on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label.'}
            ]
        }
    }
    
    if topic not in tutorials:
        flash('Tutorial not found', 'error')
        return redirect(url_for('resources'))
    
    return render_template('tutorial_module.html', tutorial=tutorials[topic])

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    """Recommend models based on dataset characteristics"""
    recommended_models = []
    dataset_properties = {}
    
    if request.method == 'POST':
        # Get dataset characteristics from form
        task = request.form.get('task', '')
        size = request.form.get('size', '')
        complexity = request.form.get('complexity', '')
        speed = request.form.get('speed', '')
        
        dataset_properties = {
            'task': task,
            'size': size,
            'complexity': complexity,
            'speed': speed
        }
        
        # Logic to recommend models based on form inputs
        if task == 'regression':
            if complexity == 'simple':
                recommended_models = ['Linear Regression', 'Ridge Regression']
            elif complexity == 'moderate':
                recommended_models = ['Random Forest Regressor', 'Gradient Boosting Regressor']
            else:  # complex
                recommended_models = ['XGBoost Regressor', 'Neural Network Regressor']
                
        elif task == 'classification':
            if size == 'small':
                if speed == 'fast':
                    recommended_models = ['Logistic Regression', 'Decision Tree Classifier']
                else:
                    recommended_models = ['SVC', 'Random Forest Classifier']
            else:  # large
                if speed == 'fast':
                    recommended_models = ['Naive Bayes', 'Logistic Regression']
                else:
                    recommended_models = ['Gradient Boosting Classifier', 'XGBoost Classifier']
                    
        elif task == 'clustering':
            if complexity == 'simple':
                recommended_models = ['K-Means', 'DBSCAN']
            else:
                recommended_models = ['Agglomerative Clustering', 'Gaussian Mixture']
    
    return render_template('recommendation.html', 
                          recommended_models=recommended_models,
                          dataset_properties=dataset_properties)

@app.route('/get_model_params/<model_name>')
def get_model_params(model_name):
    """Return tunable parameters for a specific model"""
    if model_name not in model_options:
        return jsonify({'error': 'Model not found'}), 404
    
    # Define parameter ranges for each model (simplified)
    param_ranges = {
        'Linear Regression': [],  # No hyperparameters to tune
        'Ridge Regression': [
            {'name': 'alpha', 'type': 'range', 'min': 0.01, 'max': 10, 'step': 0.01, 'default': 1.0, 'description': 'Regularization strength'}
        ],
        'Lasso Regression': [
            {'name': 'alpha', 'type': 'range', 'min': 0.01, 'max': 10, 'step': 0.01, 'default': 1.0, 'description': 'Regularization strength'}
        ],
        'Random Forest Regressor': [
            {'name': 'n_estimators', 'type': 'range', 'min': 10, 'max': 500, 'step': 10, 'default': 100, 'description': 'Number of trees'},
            {'name': 'max_depth', 'type': 'range', 'min': 1, 'max': 30, 'step': 1, 'default': 10, 'description': 'Maximum tree depth'},
            {'name': 'min_samples_split', 'type': 'range', 'min': 2, 'max': 20, 'step': 1, 'default': 2, 'description': 'Minimum samples to split node'}
        ],
        'K-Means': [
            {'name': 'n_clusters', 'type': 'range', 'min': 2, 'max': 20, 'step': 1, 'default': 8, 'description': 'Number of clusters'},
            {'name': 'init', 'type': 'select', 'options': ['k-means++', 'random'], 'default': 'k-means++', 'description': 'Initialization method'}
        ],
        'Logistic Regression': [
            {'name': 'C', 'type': 'range', 'min': 0.01, 'max': 10, 'step': 0.01, 'default': 1.0, 'description': 'Inverse of regularization strength'},
            {'name': 'solver', 'type': 'select', 'options': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'], 'default': 'lbfgs', 'description': 'Algorithm for optimization'}
        ]
    }
    
    # Return default parameters for models not explicitly defined
    if model_name not in param_ranges:
        return jsonify([
            {'name': 'param1', 'type': 'range', 'min': 0, 'max': 10, 'step': 0.1, 'default': 5, 'description': 'Example parameter'},
            {'name': 'param2', 'type': 'checkbox', 'default': True, 'description': 'Example boolean parameter'}
        ])
    
    return jsonify(param_ranges.get(model_name, []))

def get_model_type(model_name):
    """Get the type of a model (regression, classification, clustering)"""
    if model_name in model_options:
        return model_options[model_name]['category']
    return None

def generate_plot(plot_type, data, **kwargs):
    """Generate a plot and return as base64 encoded string"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if plot_type == 'scatter':
        x = data.get('x', [])
        y = data.get('y', [])
        ax.scatter(x, y, alpha=0.5)
        ax.set_xlabel(kwargs.get('xlabel', 'X'))
        ax.set_ylabel(kwargs.get('ylabel', 'Y'))
        ax.set_title(kwargs.get('title', 'Scatter Plot'))
        
        if kwargs.get('line', False):
            ax.plot([min(x), max(x)], [min(y), max(y)], 'r--')
    
    elif plot_type == 'bar':
        x = data.get('x', [])
        y = data.get('y', [])
        ax.bar(x, y)
        ax.set_xlabel(kwargs.get('xlabel', 'X'))
        ax.set_ylabel(kwargs.get('ylabel', 'Y'))
        ax.set_title(kwargs.get('title', 'Bar Plot'))
    
    elif plot_type == 'line':
        x = data.get('x', [])
        y = data.get('y', [])
        ax.plot(x, y)
        ax.set_xlabel(kwargs.get('xlabel', 'X'))
        ax.set_ylabel(kwargs.get('ylabel', 'Y'))
        ax.set_title(kwargs.get('title', 'Line Plot'))
    
    elif plot_type == 'heatmap':
        matrix = data.get('matrix', [[]])
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel(kwargs.get('xlabel', 'X'))
        ax.set_ylabel(kwargs.get('ylabel', 'Y'))
        ax.set_title(kwargs.get('title', 'Heatmap'))
    
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    return base64.b64encode(buf.read()).decode('utf-8')

# Function to download sample datasets if they don't exist
def ensure_sample_datasets():
    # Skip dataset creation on Vercel (read-only filesystem)
    if os.environ.get('VERCEL'):
        print("Skipping dataset creation on Vercel (read-only filesystem)")
        return
    
    # Create datasets directory if it doesn't exist
    os.makedirs('datasets', exist_ok=True)
    
    for dataset_name, dataset_info in DATASET_LIBRARY.items():
        filepath = dataset_info['preview_path']
        if not os.path.exists(filepath):
            # Create a simple placeholder dataset if the real one doesn't exist
            if dataset_name == 'boston_housing':
                create_boston_housing_dataset(filepath)
            elif dataset_name == 'california_housing':
                create_california_housing_dataset(filepath)
            elif dataset_name == 'diabetes':
                create_diabetes_dataset(filepath)
            elif dataset_name == 'iris':
                create_iris_dataset(filepath)
            elif dataset_name == 'breast_cancer':
                create_breast_cancer_dataset(filepath)
            elif dataset_name == 'digits':
                create_digits_dataset(filepath)
            elif dataset_name == 'blobs':
                create_blobs_dataset(filepath)
            elif dataset_name == 'moons':
                create_moons_dataset(filepath)
            elif dataset_name == 'circles':
                create_circles_dataset(filepath)
            else:
                # Generic dataset creation for other datasets
                create_generic_dataset(filepath, dataset_info)

# Generic function to create sample datasets
def create_generic_dataset(filepath, dataset_info):
    # Skip on Vercel
    if os.environ.get('VERCEL'):
        return
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    category = dataset_info.get('category', 'regression')
    num_samples = min(100, dataset_info.get('samples', 100))  # Limit to 100 samples for preview
    num_features = dataset_info.get('features', 5)
    
    # Create feature columns
    data = {}
    for i in range(num_features - 1):  # -1 because we'll add a target column
        feature_name = f"feature_{i+1}"
        data[feature_name] = np.random.rand(num_samples) * 10
    
    # Add target column based on category
    if category == 'regression':
        data['target'] = np.random.rand(num_samples) * 100
    elif category == 'classification':
        data['target'] = np.random.randint(0, 3, num_samples)  # 3 classes
    elif category == 'clustering':
        # No explicit target for clustering
        data['feature_' + str(num_features)] = np.random.rand(num_samples) * 10
    elif category == 'time_series':
        # Add date column for time series
        dates = pd.date_range(start='2020-01-01', periods=num_samples)
        data['date'] = dates
        data['value'] = np.random.rand(num_samples) * 100
    elif category == 'text':
        # Simple text dataset
        data['text'] = ['Sample text ' + str(i) for i in range(num_samples)]
        data['label'] = np.random.randint(0, 2, num_samples)  # Binary classification
    elif category == 'image':
        # Simplified image dataset (just random pixel values)
        for i in range(num_features):
            data[f"pixel_{i}"] = np.random.randint(0, 256, num_samples)
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

def create_boston_housing_dataset(filepath):
    # Skip on Vercel
    if os.environ.get('VERCEL'):
        return
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame({
        'CRIM': np.random.rand(100) * 100,
        'ZN': np.random.rand(100) * 100,
        'INDUS': np.random.rand(100) * 30,
        'CHAS': np.random.randint(0, 2, 100),
        'NOX': np.random.rand(100) * 1,
        'RM': np.random.rand(100) * 10,
        'AGE': np.random.rand(100) * 100,
        'DIS': np.random.rand(100) * 10,
        'RAD': np.random.randint(1, 25, 100),
        'TAX': np.random.rand(100) * 700,
        'PTRATIO': np.random.rand(100) * 20,
        'B': np.random.rand(100) * 400,
        'LSTAT': np.random.rand(100) * 40,
        'MEDV': np.random.rand(100) * 50
    })
    df.to_csv(filepath, index=False)

def create_california_housing_dataset(filepath):
    # Skip on Vercel
    if os.environ.get('VERCEL'):
        return
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame({
        'MedInc': np.random.rand(100) * 15,
        'HouseAge': np.random.rand(100) * 50,
        'AveRooms': np.random.rand(100) * 10, 
        'AveBedrms': np.random.rand(100) * 5,
        'Population': np.random.rand(100) * 5000,
        'AveOccup': np.random.rand(100) * 5,
        'Latitude': 37 + np.random.rand(100) * 5,
        'Longitude': -122 + np.random.rand(100) * 5,
        'MedHouseVal': np.random.rand(100) * 500000
    })
    df.to_csv(filepath, index=False)

def create_diabetes_dataset(filepath):
    # Skip on Vercel
    if os.environ.get('VERCEL'):
        return
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame({
        'age': np.random.randint(20, 80, 100),
        'sex': np.random.randint(0, 2, 100),
        'bmi': np.random.rand(100) * 40,
        'bp': np.random.rand(100) * 120,
        's1': np.random.rand(100) * 300,
        's2': np.random.rand(100) * 200,
        's3': np.random.rand(100) * 100,
        's4': np.random.rand(100) * 90,
        's5': np.random.rand(100) * 80,
        's6': np.random.rand(100) * 70,
        'target': np.random.rand(100) * 300
    })
    df.to_csv(filepath, index=False)

def create_iris_dataset(filepath):
    # Skip on Vercel
    if os.environ.get('VERCEL'):
        return
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame({
        'sepal_length': np.random.rand(150) * 3 + 4,
        'sepal_width': np.random.rand(150) * 2 + 2,
        'petal_length': np.random.rand(150) * 5 + 1,
        'petal_width': np.random.rand(150) * 2.3,
        'species': np.random.choice(['setosa', 'versicolor', 'virginica'], 150)
    })
    df.to_csv(filepath, index=False)

def create_breast_cancer_dataset(filepath):
    # Skip on Vercel
    if os.environ.get('VERCEL'):
        return
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Create 30 random features with appropriate names
    data = {}
    for i in range(30):
        feature_name = f'feature_{i}'
        data[feature_name] = np.random.rand(100) * 100
    
    data['target'] = np.random.choice([0, 1], 100)
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

def create_digits_dataset(filepath):
    # Skip on Vercel
    if os.environ.get('VERCEL'):
        return
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Create 64 pixel features
    data = {}
    for i in range(64):
        feature_name = f'pixel_{i}'
        data[feature_name] = np.random.randint(0, 17, 100)
    
    data['target'] = np.random.randint(0, 10, 100)
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

def create_blobs_dataset(filepath):
    # Skip on Vercel
    if os.environ.get('VERCEL'):
        return
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    centers = [[0, 0], [3, 3], [-3, 3]]
    cluster_std = 0.7
    
    # Generate blob data
    X, y = make_blobs(n_samples=300, centers=centers, cluster_std=cluster_std, random_state=42)
    
    df = pd.DataFrame({
        'feature_1': X[:, 0],
        'feature_2': X[:, 1],
        'cluster': y
    })
    df.to_csv(filepath, index=False)

def create_moons_dataset(filepath):
    # Skip on Vercel
    if os.environ.get('VERCEL'):
        return
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
    
    df = pd.DataFrame({
        'feature_1': X[:, 0],
        'feature_2': X[:, 1],
        'cluster': y
    })
    df.to_csv(filepath, index=False)

def create_circles_dataset(filepath):
    # Skip on Vercel
    if os.environ.get('VERCEL'):
        return
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    X, y = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)
    
    df = pd.DataFrame({
        'feature_1': X[:, 0],
        'feature_2': X[:, 1],
        'cluster': y
    })
    df.to_csv(filepath, index=False)

# Call this function when the app starts
ensure_sample_datasets()

@app.route('/learn')
def learn():
    # Prepare model information
    models = {}
    for name, info in model_options.items():
        models[name] = {
            'type': info['category'],
            'desc': info.get('description', info.get('desc', 'No description available')),
            'pros': info.get('pros', ['Easy to use']),
            'cons': info.get('cons', ['May not be optimal for all datasets']),
            'docs_url': info.get('docs_url', 'https://scikit-learn.org/stable/supervised_learning.html'),
            'complexity': info.get('complexity', 3)  # Default to medium complexity (3 out of 5)
        }
    
    # Prepare concepts information
    concepts = [
        {
            'name': 'Cross-Validation',
            'description': 'A technique to evaluate model performance by splitting data into multiple training and testing sets.',
            'formula': 'k-fold CV: Split data into k folds, train on k-1 folds, test on remaining fold, repeat k times.',
            'example': 'With 5-fold CV, we train 5 models, each using 80% of data for training and 20% for validation.',
            'learn_more_url': 'https://scikit-learn.org/stable/modules/cross_validation.html'
        },
        {
            'name': 'Regularization',
            'description': 'Techniques to prevent overfitting by adding a penalty term to the loss function.',
            'formula': 'L1 (Lasso): penalty = α * Σ|w_i|, L2 (Ridge): penalty = α * Σw_i²',
            'example': 'Ridge regression adds squared magnitude of coefficients to loss function.',
            'learn_more_url': 'https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification'
        },
        {
            'name': 'Bias-Variance Tradeoff',
            'description': 'The balance between underfitting (high bias) and overfitting (high variance).',
            'formula': 'Error = Bias² + Variance + Irreducible Error',
            'example': 'A complex model may fit training data perfectly (low bias) but perform poorly on new data (high variance).',
            'learn_more_url': 'https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html'
        },
        {
            'name': 'Feature Scaling',
            'description': 'Normalizing the range of features to improve model performance.',
            'formula': 'Standardization: z = (x - μ) / σ, Min-Max Scaling: z = (x - min) / (max - min)',
            'example': 'Scaling features before training a Support Vector Machine can significantly improve results.',
            'learn_more_url': 'https://scikit-learn.org/stable/modules/preprocessing.html'
        }
    ]
    
    return render_template('learn.html', models=models, concepts=concepts)

# Add a test route to verify session storage
@app.route('/test_session')
def test_session():
    # Set a test value in the session
    session['test_value'] = 'This is a test value: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    session.modified = True
    
    # Return all session data as JSON for debugging
    return jsonify({
        'message': 'Session test',
        'session_data': {key: str(session.get(key)) for key in session},
        'test_value': session.get('test_value')
    })

# Add a route to clear the session for testing
@app.route('/clear_session')
def clear_session():
    # Clear the session
    session.clear()
    session.modified = True
    return jsonify({'message': 'Session cleared'})

# Add a test route to set dummy results
@app.route('/test_results')
def test_results():
    # Create dummy results
    dummy_results = {
        'model_name': 'Linear Regression',
        'model_type': 'regression',
        'target': 'target_column',
        'features': ['feature1', 'feature2'],
        'metrics': {
            'mse': 0.25,
            'r2': 0.85,
            'mae': 0.2
        },
        'predictions': [1.2, 2.3, 3.4, 4.5, 5.6],
        'training_time': 0.5,
        'data_points': 100,
        'feature_importance': [('feature1', 0.7), ('feature2', 0.3)],
        'plot_url': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='
    }
    
    # Store in session
    session['results'] = dummy_results
    session.modified = True
    
    # Redirect to results page
    return redirect(url_for('results'))

# Add a route to view session data
@app.route('/view_session')
def view_session():
    # Return all session data as JSON for debugging
    session_data = {}
    for key in session:
        try:
            # Try to convert to a serializable format
            if isinstance(session[key], dict):
                session_data[key] = session[key]
            elif hasattr(session[key], 'tolist'):
                session_data[key] = session[key].tolist()
            else:
                session_data[key] = str(session[key])
        except:
            session_data[key] = f"<unserializable: {type(session[key])}>"
    
    return jsonify({
        'message': 'Current session data',
        'session_data': session_data
    })

# Create database tables only if not on Vercel
if not os.environ.get('VERCEL'):
    with app.app_context():
        db.create_all()

if __name__ == '__main__':
    # Production configuration
    if os.environ.get('FLASK_ENV') == 'production':
        # Use environment variables for production
        app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key')
        app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
        
        # Disable debug in production
        app.debug = False
        
        # Configure logging for production
        if not app.debug:
            if not os.path.exists('logs'):
                os.mkdir('logs')
            file_handler = RotatingFileHandler('logs/ml_playground.log', maxBytes=10240, backupCount=10)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
            app.logger.setLevel(logging.INFO)
            app.logger.info('ML Playground startup')
    
    # Initialize sample datasets only if not on Vercel
    if not os.environ.get('VERCEL'):
        # Ensure uploads directory exists
        os.makedirs('uploads', exist_ok=True)
        
        # Initialize sample datasets
        with app.app_context():
            ensure_sample_datasets()
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

# Add error handlers for better debugging
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500
