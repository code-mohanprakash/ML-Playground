# ML Playground

A comprehensive Machine Learning web application that allows users to upload datasets, train models, and analyze results without writing any code.

## Features

- **Dataset Management**: Upload your own CSV files or use pre-built sample datasets
- **Multiple ML Models**: Support for Regression, Classification, and Clustering algorithms
- **Interactive Results**: Visualize model performance with charts and metrics
- **Model Comparison**: Compare different models side-by-side
- **Hyperparameter Tuning**: Optimize model parameters for better performance
- **Learning Resources**: Educational content and tutorials
- **Professional UI**: Clean, enterprise-ready interface

## Tech Stack

- **Backend**: Flask (Python)
- **Database**: SQLite (with PostgreSQL support for production)
- **ML Libraries**: Scikit-learn, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML, CSS (Tailwind), JavaScript
- **Deployment**: Vercel-ready

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/code-mohanprakash/ML-Playground.git
   cd ML-Playground
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   ```bash
   export FLASK_ENV=development
   export SECRET_KEY=your-secret-key-here
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:5000`

### Production Deployment

#### Vercel Deployment

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Deploy to Vercel**
   ```bash
   vercel
   ```

3. **Set environment variables in Vercel dashboard**
   - `SECRET_KEY`: Your secret key
   - `FLASK_ENV`: production

#### Other Platforms

The app is compatible with:
- Heroku
- Railway
- DigitalOcean App Platform
- Any platform supporting Python/Flask

## Project Structure

```
ML-Playground/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── vercel.json           # Vercel configuration
├── runtime.txt           # Python runtime version
├── templates/            # HTML templates
├── static/              # CSS, JS, images
├── datasets/            # Sample datasets
├── uploads/             # User uploaded files
├── logs/                # Application logs
└── models/              # Database models
```

## Available Models

### Regression
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regression
- K-Neighbors Regressor
- Decision Tree Regressor
- Bayesian Ridge

### Classification
- Logistic Regression
- Random Forest Classifier
- Support Vector Classifier
- K-Neighbors Classifier
- Decision Tree Classifier
- Gradient Boosting Classifier
- AdaBoost Classifier
- Extra Trees Classifier
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Bernoulli Naive Bayes
- Linear Discriminant Analysis
- Quadratic Discriminant Analysis

### Clustering
- K-Means
- Agglomerative Clustering
- DBSCAN
- Mean Shift
- Spectral Clustering
- Gaussian Mixture Model
- Birch Clustering

## Sample Datasets

The application includes 35+ pre-built datasets across different categories:

- **Regression**: Boston Housing, California Housing, Diabetes, Bike Sharing, Wine Quality, etc.
- **Classification**: Iris, Breast Cancer, Wine, Heart Disease, Credit Card Fraud, etc.
- **Clustering**: Various synthetic datasets for clustering analysis

## API Endpoints

- `GET /` - Home page
- `GET /upload` - Upload dataset page
- `POST /upload_file` - File upload endpoint
- `GET /select_model` - Model selection page
- `POST /run_model` - Model training endpoint
- `GET /results` - View model results
- `GET /datasets` - Browse sample datasets
- `GET /learn` - Learning resources
- `GET /resources` - Additional resources

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, email support@mlplayground.com or create an issue in the GitHub repository.

## Acknowledgments

- Scikit-learn team for the excellent ML library
- Flask team for the web framework
- Tailwind CSS for the styling framework
- All contributors and users of this project 