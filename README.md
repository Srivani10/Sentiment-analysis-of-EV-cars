# Sentiment-analysis-of-EV-cars
Sentiment analysis of four-wheeler EV user reviews using NLP and ML models
# EV Car Sentiment Analysis

Project Overview
This project performs **sentiment analysis** on user reviews of **four-wheeler electric vehicles (EVs)** in India. The goal is to classify user sentiments as **Positive, Neutral, or Negative** using **Natural Language Processing (NLP) and Machine Learning (ML) models**.

## 📂 Dataset
- **Source:** Kaggle (originally scraped from CarDekho and CarWale)
- **Features:**
  - `Review Text` - Customer feedback
  - `Rating` - Numerical rating given by users
  - `Sentiment` - Derived sentiment label (Positive/Neutral/Negative)

🔧 Tech Stack & Libraries Used
- Python: pandas, numpy, re, nltk, textblob, BeautifulSoup
- ML Models: Logistic Regression, Random Forest, Decision Tree, Naïve Bayes
- Data Processing: SMOTE (for handling class imbalance), CountVectorizer
- **Visualization**: Matplotlib

## 📊 Project Workflow
1. **Data Preprocessing**:
   - Clean text (remove HTML, stopwords, special characters, lemmatization)
   - Convert numerical ratings to sentiment labels
2. **Exploratory Data Analysis (EDA)**:
   - Visualize sentiment distribution
3. **Feature Engineering & Model Training**:
   - Convert text to numerical vectors using CountVectorizer
   - Train ML models to classify sentiments
4. **Model Evaluation**:
   - Measure accuracy, precision, recall, and F1-score

## 🛠 Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/EV-Sentiment-Analysis.git
   cd EV-Sentiment-Analysis
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. Run Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
   Open `EV_car_sentiment_analysis.ipynb` and run the cells.

 Key Findings
- **Majority of sentiments were positive**, highlighting satisfaction with **design and mileage**.
- **Negative sentiments** mainly concerned **pricing and charging infrastructure**.
- **Random Forest and Logistic Regression** performed best for sentiment classification.

Future Improvements
- Experiment with **TF-IDF and Word Embeddings** for better feature extraction.
- Try **deep learning models** like LSTMs or Transformers for improved accuracy.
- Expand dataset to include **more brands and user demographics**.



