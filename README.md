# AI-Powered Personalized Job Recommendation System: A ML Approach

An advanced machine learning and natural language processing (NLP) capstone project designed to revolutionize the recruitment process by matching job seekers with ideal job categories based on their resumes, skills, experience, and career goals[cite: 103, 116].

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features & Architecture](#-key-features--architecture)
- [Dataset & Methodology](#-dataset-methodology)
- [Model Performance & Benchmarking](#-model-performance-benchmarking)
- [Streamlit Web Application](#-streamlit-web-application)
- [Repository Structure](#-repository-structure)
- [Tech Stack & Installation](#-tech-stack--installation)
- [Limitations & Future Improvements](#-limitations--future-improvements)
- [Author & Project Metadata](#-author-project-metadata)

---

## 📖 Project Overview
Traditional recruitment platforms rely heavily on simplistic keyword matching or collaborative filtering, often overlooking nuanced interpersonal skills, professional objectives, and contextual capabilities[cite: 103, 116]. 

This project bridges the gap by building an intelligent recommendation engine trained on a large-scale dataset of job postings[cite: 116]. Using advanced NLP feature extraction (TF-IDF) and a robust **Stacking Ensemble Model**, the system automatically processes unstructured resume text and predicts the most suitable job category with high precision[cite: 116, 127].

---

## ⚙️ Key Features & Architecture
* **Advanced NLP Pipeline:** Cleans raw resume text (removing URLs, special characters, non-ASCII text, and stopwords) and transforms text features using Term Frequency-Inverse Document Frequency (`TfidfVectorizer`)[cite: 126, 127].
* **Stratified Sampling:** Efficiently scales a massive 1.61 million row dataset down to a representative sample of 100,000 rows while preserving class proportions across job titles[cite: 120, 121].
* **Stacking Ensemble Model:** Combines Random Forest, XGBoost, and LightGBM base learners with a Logistic Regression meta-classifier to achieve superior predictive accuracy[cite: 128, 150].
* **Interactive Streamlit Web App:** Eliminates technical barriers by providing a clean graphical user interface where users can upload their CV (`.txt` or `.pdf`) and receive instant job recommendations[cite: 132, 134].

---

## 📊 Dataset & Methodology
* **Source:** Kaggle job listings dataset (Licensed under CC0: Public Domain, with a usability score of 10/10)[cite: 121].
* **Data Preprocessing & Feature Engineering:**
  * Extracted minimum and maximum ranges for experience and salary[cite: 124].
  * One-hot encoded categorical variables (Work Type, Salary Bracket, Sector, Industry) while dropping the first category to avoid multicollinearity[cite: 124, 125].
  * Mapped 147 granular job titles into 48 distinct standardized job categories[cite: 120, 125].
  * Label-encoded the target `Category` column[cite: 125].

---

## 🏆 Model Performance & Benchmarking

The system evaluated 8 different machine learning classification models using 5-fold cross-validation and rigorous evaluation metrics[cite: 128, 130, 150]. The **Stacking Ensemble Model** emerged as the undisputed champion[cite: 131]:

| Model | Accuracy Score | Precision Score | Recall Score | F1 Score | ROC-AUC Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Neighbors (OneVsRest)**[cite: 131] | 0.658[cite: 131] | 0.60[cite: 131] | 0.65[cite: 131] | 0.62[cite: 131] | 0.69[cite: 131] |
| **Decision Tree Classifier**[cite: 131] | 0.691[cite: 131] | 0.65[cite: 131] | 0.68[cite: 131] | 0.66[cite: 131] | 0.72[cite: 131] |
| **Support Vector Classifier (SVC)**[cite: 131] | 0.769[cite: 131] | 0.73[cite: 131] | 0.76[cite: 131] | 0.74[cite: 131] | 0.81[cite: 131] |
| **Random Forest Classifier**[cite: 131] | 0.776[cite: 131] | 0.74[cite: 131] | 0.77[cite: 131] | 0.75[cite: 131] | 0.835[cite: 131] |
| **Logistic Regression**[cite: 131] | 0.796[cite: 131] | 0.78[cite: 131] | 0.79[cite: 131] | 0.78[cite: 131] | 0.86[cite: 131] |
| **Gradient Boosting Classifier**[cite: 131] | 0.849[cite: 131] | 0.83[cite: 131] | 0.85[cite: 131] | 0.84[cite: 131] | 0.90[cite: 131] |
| **XGBoost Classifier**[cite: 131] | 0.864[cite: 131] | 0.85[cite: 131] | 0.86[cite: 131] | 0.86[cite: 131] | 0.924[cite: 131] |
| **Ensemble (RF, XGB, LGBM)** *(Champion)*[cite: 131] | **0.916**[cite: 131] | **0.91**[cite: 131] | **0.92**[cite: 131] | **0.92**[cite: 131] | **0.95**[cite: 131] |

---

## 💻 Streamlit Web Application

To make the champion model accessible to end-users without requiring manual Python scripting, a Streamlit web application was developed[cite: 132, 134].

### How to Run the Web App Locally:
1. Ensure all required pickle files (`stacking_model.pkl`, `clf.pkl`, `tfidf.pkl`) and the application script (`jobrecapp.py`) are in the same working directory[cite: 132, 133].
2. Install dependencies:
   ```bash
   pip install streamlit pandas scikit-learn xgboost lightgbm nltk

```

3. Run the Streamlit application:
```bash
streamlit run jobrecapp.py

```


4. Open the local URL (`http://localhost:8501`), upload your resume (`.txt` or `.pdf`), and view your predicted career category within seconds.



---

## 📁 Repository Structure

```text
├── Report.pdf               # Comprehensive 42-page capstone report[cite: 114]
├── Ppt.pptx                 # Presentation slides deck[cite: 101]
├── Project proposal.pdf     # Initial project proposal[cite: 102]
├── Dataset.csv              # Supporting dataset export[cite: 97]
├── Webapp.py                # Streamlit web application source code[cite: 133]
├── stacking_model.pkl       # Trained Stacking Ensemble model pickle file[cite: 132]
├── clf.pkl                  # Classifier model pickle file[cite: 132, 133]
├── tfidf.pkl                # Trained TF-IDF vectorizer pickle file[cite: 132, 133]
└── README.md                # Project documentation

```

---

## 🛠️ Tech Stack & Dependencies

* **Programming Language:** Python 3


* **Data Manipulation & Analysis:** `pandas`, `numpy`


* **Data Visualization:** `matplotlib`, `seaborn`


* **Machine Learning & Modeling:** `scikit-learn`, `XGBoost`, `LightGBM`


* **Natural Language Processing:** `NLTK`, `TfidfVectorizer`


* **Web Deployment:** `Streamlit`



---

## ⚠️ Limitations & Future Directions

* **Algorithm Bias:** Historical training data may inadvertently embed biases that impact minority groups, requiring ongoing fairness audits.


* **Static Dataset:** The model relies on a stationary Kaggle dataset rather than real-time streaming job portals (e.g., LinkedIn/Indeed APIs).


* **Computational Constraints:** Training complex ensemble and transformer-based models (like BERT) demands high-performance GPU infrastructure.



---

## 👥 Author & Project Metadata

* **Author:** Mayank Panchal (Student ID: `7796043`)


* **Institution:** University of Wollongong, Dubai


* **Module / Capstone:** BALA302 Capstone Project


* **Date:** 21 March 2025



```

```
