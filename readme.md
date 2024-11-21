# Text Classification for IMDB Movie Reviews


### Table of Contents
1. [Project Overview](#project-overview)
2. [Project Objectives](#project-objectives)
3. [Dataset Description](#dataset-description)
4. [Project Workflow](#project-workflow)
   - [Data Cleaning and Preprocessing](#1-data-cleaning-and-preprocessing)
   - [Feature Extraction with TF-IDF](#2-feature-extraction-with-tf-idf)
   - [Model Training](#3-model-training)
   - [Model Evaluation](#4-model-evaluation)
5. [Results and Insights](#results-and-insights)
6. [Usage](#usage)
   - [Prerequisites](#prerequisites)
   - [Running the Project](#running-the-project)
   - [Example Code Snippet](#example-code-snippet)
7. [Project Structure](#project-structure)
8. [Future Improvements](#future-improvements)
9. [Contact](#contact)

---


### Project Overview
This project implements a text classification system to categorize IMDB movie reviews as either *positive* or *negative*. The focus is on creating a model optimized for limited computational resources, using classical machine learning models and a simple neural network. This project compares model performance to identify the most effective approach.

### Project Objectives
- Use the IMDB Movie Reviews dataset for binary sentiment classification.
- Implement data cleaning and preprocessing techniques.
- Create TF-IDF vectorization for feature extraction.
- Train and compare multiple classical ML models (Naive Bayes, Logistic Regression, SVM).
- Implement a neural network using Keras for sentiment classification.
- Perform hyperparameter tuning and k-fold cross-validation.
- Evaluate model performance using confusion matrices and ROC curves.

---

## Dataset Description
**Dataset**: [IMDB Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), a binary sentiment dataset with 50,000 reviews labeled as positive or negative.  
- **Positive Reviews**: Score ≥ 7 out of 10
- **Negative Reviews**: Score ≤ 4 out of 10

---

## Project Workflow

### 1. Data Cleaning and Preprocessing
- Removed HTML tags, non-alphabetic characters, and stop words.
- Applied lemmatization to convert words to their base forms.
- Generated word clouds to visualize the most common words in positive and negative reviews.

### 2. Feature Extraction with TF-IDF
- Used TF-IDF vectorization to transform text into a feature matrix.
- Limited features to the top 5,000 words to optimize for memory and processing efficiency.

### 3. Model Training
- Trained and evaluated the following models:
  - **Naive Bayes**
  - **Logistic Regression**
  - **Support Vector Machine (SVM)**
  - **Simple Neural Network (Keras)**

### 4. Model Evaluation
- Evaluated models using accuracy, precision, recall, F1-score, and AUC.
- Created confusion matrices and ROC curves for detailed performance insights.
- Performed hyperparameter tuning using GridSearchCV and k-fold cross-validation.

---

## Results and Insights
| Model               | Accuracy | Precision | Recall | F1-Score | AUC   |
|---------------------|----------|-----------|--------|----------|-------|
| Naive Bayes         | 0.8469   | 0.8449    | 0.8527 | 0.8488   | 0.9233|
| Logistic Regression | 0.8798   | 0.8686    | 0.8972 | 0.8827   | 0.9524|
| SVM                 | 0.8845   | 0.8719    | 0.9036 | 0.8874   | 0.9555|
| Neural Network      | 0.8716   | 0.8566    | 0.8950 | 0.8754   | 0.9427|

**Best Model**: SVM demonstrated the highest scores across precision, recall, F1-Score, and AUC, making it the most effective model for this dataset.

### Key Insights
- **SVM** provided the best overall performance, particularly in handling complex text patterns.
- **Logistic Regression** and the **Neural Network** also performed well and achieved good balance across metrics.
- **Naive Bayes** showed strong performance but was slightly behind in comparison with the other models.

---

## Usage

### Prerequisites
- Python 3.7+
- Install the required libraries:
  ```bash
  pip install pandas numpy matplotlib seaborn spacy scikit-learn tensorflow tabulate wordcloud




## Running the Project

**1. Load the Dataset**
- Place the IMDB dataset CSV file in the project folder, and load it using Pandas:
  ```bash 
  import pandas as pd
  df = pd.read_csv('IMDB Dataset.csv')

**2. Clean and Preprocess Text Data**
- Use the clean_review function to preprocess each review by removing unnecessary characters and lemmatizing words.

**3. Train Models**
- Apply TF-IDF vectorization, then train models including Naive Bayes, Logistic Regression, and SVM.

**3. Evaluate and Compare Models**
- Generate confusion matrices, ROC curves, and compare performance metrics.

## Example Code Snippet

- ### Example: Training SVM Model
  ```bash
  from sklearn.svm import SVC
  svm = SVC()
  svm.fit(X_train, y_train)
  svm_pred = svm.predict(X_test)

  from sklearn.metrics import accuracy_score
  print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

## Project Structure
- nlp_02.py: Main code file with data loading, preprocessing, model training, and evaluation.
- README.md: Project documentation.
- svm_model.pkl: Saved SVM model file.
- IMDB Dataset.csv: IMDB dataset file used for training and evaluation.

## Future Improvements
- Experiment with more advanced NLP models, such as Transformer-based models (e.g., BERT) for improved accuracy.
- Implement additional data cleaning steps, like removing rare words or applying further regularization.
- Try ensemble methods combining multiple models for potentially better results.

## Contact
For questions or collaboration, please reach out at 📧 aydanrzyv@gmail.com.

