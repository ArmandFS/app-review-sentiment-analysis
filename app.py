import streamlit as st
import pickle
import pandas as pd
from preprocessing import preprocess_text
from sklearn.linear_model import LogisticRegression

import nltk
nltk.download('punkt')  
nltk.download('stopwords') 


#static images
logo = 'Crunchyroll_Logo.png'
polarity_pie_chart = 'pie_chart.png'
model_accuracy_bar_chart = 'model_comparison.png'
accuracy_heatmap = 'heatmap.png'


with open("tfidf_vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)


with open("logistic_regression_model.pkl", "rb") as model_file:
    logistic_regression_model = pickle.load(model_file)

#Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Pages", ["About", "Analytics","Sentiment Analysis"])


if page == "About":
    st.image(logo, width=100)
    st.title("Crunchyroll Review Sentiment Analysis")
    st.subheader("⭐ Project Overview ⭐")
    st.markdown("""
    This project analyzes the sentiment of reviews for the **Crunchyroll app** on the Google Play Store.
    Using Natural Language Processing (NLP) and Machine Learning, we aim to classify reviews as **positive**, **neutral**, or **negative**.
    
    ### ✨ Objectives ✨
    - **Understand user sentiment**: To understand how to create a sentiment analysis model and implement it to streamlit.
    - **Drive business decisions**: Listening to user feedback, criticism, and improving the app.
    
    ### 🚀 Tech Stack 🚀
    - **Python** for computation and modeling.
    - **Streamlit** for an interactive web application.
    - **Scikit-Learn** for logistic regression, random forest and TF-IDF vectorization.
    - **NLTK** for text preprocessing.
    
    ### 🙂 Made by Me! 🙂
    - Made by **ArmandFS**
    - Follow me on Github! https://github.com/ArmandFS
    - Connect with me on LinkedIn! https://www.linkedin.com/in/armandfs/
    ---

    **Use the navigation panel on the left to switch to the Sentiment Analysis page and try it out!**
    """)


elif page == "Analytics":
    st.title("📊 Analytics 📊")
    
    #pie chart plotting code
    st.header("Sentiment Polarity Distribution")
    st.markdown("This pie chart displays the distribution of sentiment polarities across Crunchyroll app reviews.")
    with st.spinner('Loading pie chart...'):
        show_pie_chart = st.button("Show Sentiment Polarity Pie Chart")
        if show_pie_chart:
            st.image(polarity_pie_chart, use_column_width=True, width=200)
    
    st.markdown("""
        **Insights**:
        - This pie chart visualizes the proportions of positive, neutral, and negative reviews. 
        - This illustrates that 46.5% of users have a positive sentiment towards the application, 29.8% of users have a   neutral review, and the remaining 23.7% have a negative sentiment of the application. 
    """)
    
    #Model Accuracy Bar Chart
    st.header("Model Accuracy Comparison")
    st.markdown("This bar chart compares the training and test accuracies of the Random Forest and Logistic Regression models.")
    
    with st.spinner('Loading bar chart...'):
        show_accuracy_bar_chart = st.button("Show Model Accuracy Bar Chart")
        if show_accuracy_bar_chart:
            st.image(model_accuracy_bar_chart, use_column_width=True, width=700)
    st.markdown("""
        **Insights**:
        - This bar chart provides insight into how well each model performed on training and test data.
        - The random forest has perfect 100% training accuracy but a mediocre 83% accuracy on testing. This may indicate overfitting which can be fixed by cleaning up the data or tuning more hyperparameters on the random forest model.
        - The logistic regressin both have great training and testing accuracy, indicating that tests great on unseen data. Further hyperparameter tuning may bump the accuracy above 95% if done correctly. 
    """)

    #Model Accuracy Heatmap
    st.header("Model Accuracy Heatmap")
    st.markdown("This heatmap provides a visual representation of the accuracy of each model on training and test data.")
    
    with st.spinner('Loading heatmap...'):
        show_heatmap = st.button("Show Accuracy Heatmap")
        if show_heatmap:
            st.image(accuracy_heatmap, use_column_width=True, width=350)
    
    
    st.markdown("""
        **Insights**:
        - The heatmap displays a more compact view of accuracy scores across models and data types.
        - Lighter shades indicate higher accuracy, and displays the same numerical values as the bar chart.
    """)
else:
    st.title("🚀 Sentiment Analysis Tool 🚀")
    st.markdown("""
    ### Analyze User Sentiments
    Enter any text below to classify its sentiment as **Positive**, **Neutral**, or **Negative**.
    Think of this as reviewing the actual CrunchyRoll application, what you hate, or like about it. CTRL + Enter to see sentiment output.
    """)

   
    user_input = st.text_area("Type your review here:", "", height=160)

    if user_input:
        processed_text = preprocess_text(user_input)
        text_features = tfidf.transform([processed_text])
        prediction = logistic_regression_model.predict(text_features)[0]
        
        if prediction == "positive":
            sentiment = "🙂 Positive 🙂"
            color = "green"
            explanation = "The review expresses a favorable opinion about the application, highlighting positive experiences or satisfaction with features."
        elif prediction == "negative":
            sentiment = "😠 Negative 😠"
            color = "red"
            explanation = "The review shows a critical perspective, possibly highlighting issues, frustrations, or not meeting expectations with the application."
        else:
            sentiment = "😐 Neutral 😐"
            color = "orange"
            explanation = "The review provides a balanced view, neither liking nor disliking the application, and might include both positive and negative aspects."

        
        st.markdown(
            f"<h2 style='text-align: center; color: {color};'>Sentiment: {sentiment}</h2>", 
            unsafe_allow_html=True
        )
      
        st.markdown(
            f"<p style='text-align: center; color: #555555; font-size: 18px;'>{explanation}</p>",
            unsafe_allow_html=True
        )

    #Additional CSS for custom styling
    st.markdown("""
        <style>
        /* Sidebar styling */
        .sidebar .sidebar-content { background-color: #333333; color: white; }

        /* Title and headers styling */
        h1, h2, h3, .sidebar .sidebar-content h2 {
            font-family: 'Arial', sans-serif;
        }

        /* Text area and input styling */
        .stTextInput, .stTextArea {
            border: 10px ;
            padding: 10px;
            border-radius: 5px;
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-thumb { background: #1f77b4; border-radius: 4px; }

        /* Animation effects */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* Hover effects */
        .stButton:hover {
            background-color: #f7c1b4;
            transition: all 0.3s ease;
        }

        </style>
    """, unsafe_allow_html=True)

