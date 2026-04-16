import streamlit as st
import joblib

# 1. Set up the page design
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎭")

st.title("🎭 Amazon Review Sentiment Analyzer")
st.write("This application is powered by a custom Machine Learning model trained on over 24,000 real customer reviews.")

st.markdown("---")

# 2. Load the trained model into the app's memory
# The @st.cache_resource command ensures the model only loads once, keeping the app fast
@st.cache_resource
def load_model():
    try:
        # Make sure the filename matches exactly what you saved!
        return joblib.load('my_sentiment_model.pkl')
    except FileNotFoundError:
        st.error("🚨 Error: 'my_sentiment_model.pkl' not found. Please ensure it is in the same folder as this app.py file.")
        st.stop()

model = load_model()

# 3. Create the User Interface
st.subheader("Test the Model")
user_input = st.text_area("Enter a fake product review below to test its accuracy:", 
                          placeholder="e.g., I absolutely loved this, it works perfectly!", 
                          height=150)

# 4. Handle the Prediction when the user clicks the button
if st.button("Analyze Sentiment", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing tone..."):
            # The model makes its prediction
            prediction = model.predict([user_input])[0]
            
            # Display the results with custom styling
            st.markdown("### Model Prediction:")
            if prediction == 'positive':
                st.success("✨ **POSITIVE** 😃")
                st.write("The model detected a favorable or happy emotional tone.")
            elif prediction == 'negative':
                st.error("⚠️ **NEGATIVE** 😠")
                st.write("The model detected frustration, anger, or dissatisfaction.")
            else:
                st.info("📊 **NEUTRAL** 😐")
                st.write("The model detected a factual or emotionally neutral tone.")