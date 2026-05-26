# Fake-News-Detector-App
This is a machine learning–powered web application that detects whether a given news text is fake or real using NLP and classification models. The app is built with Python, Streamlit, and scikit‑learn, and provides a clean, interactive interface for users to paste news content and instantly see a prediction.

Project Overview
Fake news has become a major problem in the digital age, so this project aims to:

Train a classifier model on labeled news‑article data.
Allow users to input a news snippet or article.
Predict whether the news is likely fake or real and show the result in a user‑friendly UI.

The model is trained on a dataset of real and fake news headlines or articles, and uses techniques like TF‑IDF vectorization followed by a classifier (e.g., Logistic Regression, Naive Bayes, or SVM) to make predictions. The same trained model is loaded inside the Streamlit app to give real‑time results.


Features:
Text input: Enter any news text and get a prediction.
Simple UI: Clean, minimal interface built with Streamlit.
Fast prediction: Real‑time classification once the model is loaded.
Extensible: Easy to swap in different models or datasets.

🧠 Model Details
Algorithm: Logistic Regression
Vectorization: TF-IDF (stop words removed, max_df = 0.7)
Train-Test Split: 80% training, 20% testing
Max Iterations: 1000

📊 Results
Accuracy: 94-96% accuracy
Evaluation:
    Accuracy Score
    Confusion Matrix

Technologies Used:
Python – Core programming language
Streamlit – Interactive web app framework
scikit‑learn – For TF‑IDF vectorizer and classifier
Pandas, NumPy – For data handling and preprocessing
joblib or pickle – For saving and loading the trained model


## 📂 Project Structure
Fake-News-Detector-App/
│── app.py                 # Streamlit application
│── fake_news_model.pkl    # Trained ML model
│── vectorizer.pkl         # TF-IDF vectorizer
│── dataset.csv            # Dataset used for training
│── fake_and_real_news.py  # Model training script
│── requirements.txt       # Dependencies
│── README.md              # Project documentation


▶️ How to Run:
1.Clone the repository
2.Install dependencies
3.pip install -r requirements.txt
4.Run the app
5.streamlit run app.py


🌐 Live Demo
https://fake-news-detector-app-ermiva5zjpsgfgc8y6ky6y.streamlit.app/

⚠️ Limitations
Model only analyzes text patterns
Does not verify real-world facts
Accuracy depends on dataset quality


💡 Future Improvements
Add confidence score
Improve accuracy using advanced NLP models
Enhance UI/UX

