import streamlit as st
import joblib
import re   

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# -------------------------
# Simple UI Styling
# -------------------------
st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.stTextArea textarea {
    background-color: #1e1e1e;
    color: white;
    border-radius: 10px;
    padding: 10px;
}

.stButton button {
    background-color: #ff4b4b;
    color: white;
    font-weight: bold;
    border-radius: 8px;
}

.result-box {
    padding:15px;
    border-radius:10px;
    margin-top:10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load("fake_news_model.pkl")

@st.cache_resource
def load_vectorizer():
    return joblib.load("vectorizer.pkl")

model = load_model()
vectorizer = load_vectorizer()

# -------------------------
# Text Cleaning
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("📌 About")
st.sidebar.info(
"""
Fake News Detection App

This project uses:
• Machine Learning  
• NLP (TF-IDF Vectorization)  
• Streamlit for UI
"""
)

# -------------------------
# Main UI
# -------------------------
st.title("📰 Fake News Detection System")

st.write("Paste a news article below and the AI will predict whether it is **Fake or Real**.")

news = st.text_area("Enter News Article")

# -------------------------
# Prediction Button
# -------------------------
if st.button("Predict"):

    if news.strip() != "":

        # -------------------------
        # News Statistics
        # -------------------------
        word_count = len(news.split())
        char_count = len(news)

        st.subheader("📝 News Statistics")

        col1, col2 = st.columns(2)

        col1.metric("Word Count", word_count)
        col2.metric("Character Count", char_count)

        # -------------------------
        # Clean Text 
        # -------------------------
        cleaned_news = clean_text(news)

        # -------------------------
        # Vectorization
        # -------------------------
        news_vector = vectorizer.transform([cleaned_news])

        # -------------------------
        # Prediction
        # -------------------------
        prediction = model.predict(news_vector)

        
        st.write("Prediction:", prediction[0])

        st.subheader("📢 Prediction Result")

        if prediction[0] == 1:
            st.success("✅ This looks like Real News")
        else:
            st.error("🚨 This looks like Fake News")

        # -------------------------
        # Confidence Score
        # -------------------------
        if hasattr(model, "predict_proba"):

            proba = model.predict_proba(news_vector)

            fake_prob = proba[0][0]
            real_prob = proba[0][1]

            st.subheader("📊 Prediction Confidence")

            confidence = int(max(fake_prob, real_prob) * 100)

            st.progress(confidence)

            col1, col2 = st.columns(2)

            col1.metric("Fake Probability", str(round(fake_prob*100,2)) + "%")
            col2.metric("Real Probability", str(round(real_prob*100,2)) + "%")

        # -------------------------
        # Important Words
        # -------------------------
        feature_names = vectorizer.get_feature_names_out()

        vector_array = news_vector.toarray()[0]

        top_indices = vector_array.argsort()[-10:][::-1]

        important_words = [feature_names[i] for i in top_indices]

        st.subheader("🔍 Important Words Influencing Prediction")

        st.info(", ".join(important_words))

    else:
        st.warning("⚠ Please enter text")
