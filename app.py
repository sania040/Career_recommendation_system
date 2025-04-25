import streamlit as st
from sklearn.inspection import permutation_importance
import pandas as pd
from logic import (
    calculate_overall_score,
    train_ensemble_model,
    recommend_career_with_rf,
    explain_recommendations
)
import shap
import matplotlib.pyplot as plt

DATA_PATH = "data/careers_data.csv"

@st.cache_data
def load_data(path):
    """Load career dataset."""
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Load data and train model
data = load_data(DATA_PATH)
if data is not None:
    model, features = train_ensemble_model(data)

    # Streamlit App
    st.title("🎓 Enhanced Career Path Explorer")
    st.markdown("Discover your ideal career path based on your unique traits and aptitudes.")

    # Tabs for clean layout
    tab1, tab2, tab3 = st.tabs(["📝 Questionnaire", "📊 Results", "🔍 Explanation"])

    # Tab 1: Questionnaire
    with tab1:
        st.header("📝 Tell Us About Yourself")
        st.markdown("Rate the following statements and skills:")

        personality_questions = [
            "I enjoy exploring new ideas and experiences.",
            "I am well-organized and detail-oriented.",
            "I feel energized when interacting with others.",
            "I enjoy helping others and value their feelings.",
            "I often feel anxious or stressed in challenging situations."
        ]
        aptitude_questions = [
            "How would you rate your numerical skills?",
            "How would you rate your spatial reasoning skills?",
            "How would you rate your ability to recognize patterns?",
            "How would you rate your abstract reasoning skills?",
            "How would you rate your verbal communication skills?"
        ]

        personality_responses = [
            st.radio(q, ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"])
            for q in personality_questions
        ]
        aptitude_responses = [
            st.radio(q, ["Very Poor", "Poor", "Average", "Good", "Excellent"])
            for q in aptitude_questions
        ]

        if st.button("Submit"):
            user_scores = calculate_overall_score(personality_responses, aptitude_responses)
            st.session_state["user_scores"] = user_scores
            st.success("Responses recorded! Check the Results tab.")

    # Tab 2: Results
    with tab2:
        st.header("📊 Your Personalized Career Recommendations")

        if "user_scores" in st.session_state:
            user_scores = st.session_state["user_scores"]
            recommendations = recommend_career_with_rf(user_scores, model, data, features)

            st.markdown("Based on your responses, here are some careers that might suit you best:")
            for i, (career, score) in enumerate(recommendations, 1):
                st.write(f"**{i}. {career}** — Confidence Level: {score:.0%}")

            st.markdown("""
                _These suggestions are tailored to match your unique skills and preferences. 
                Explore them further to see how they align with your aspirations!_
            """)
        else:
            st.warning("Complete the questionnaire first to see your personalized results.")

    # Tab 3: Explanation
    with tab3:
        st.header("🔍 Why These Recommendations?")

        if "user_scores" in st.session_state:
            user_scores = st.session_state["user_scores"]
            user_input_df = pd.DataFrame([user_scores])

            try:
                # Align features for the model
                user_input_df_aligned = user_input_df[features]

                st.subheader("🌟 Key Strengths That Shaped Your Recommendations")

                # Feature importance
                feature_importances = model.named_steps["rf"].feature_importances_
                feature_importance_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": feature_importances
                }).sort_values(by="Importance", ascending=False)

                # Extract top 3 features
                top_features = feature_importance_df.head(3)
                
                # User-friendly explanations
                user_strengths = {
                    "O_score": "creativity and openness to experience",
                    "C_score": "organization and attention to detail",
                    "E_score": "enthusiasm and social skills",
                    "A_score": "compassion and teamwork",
                    "N_score": "resilience and emotional awareness",
                    "Numerical Aptitude": "number-crunching and analytical thinking",
                    "Spatial Aptitude": "visual and spatial reasoning",
                    "Verbal Reasoning": "communication and language skills"
                }

                # Dynamic explanation
                for _, row in top_features.iterrows():
                    feature = row["Feature"]
                    feature_name = user_strengths.get(feature, feature.replace('_', ' '))
                    st.markdown(f"- Your **{feature_name}** strongly influenced these career recommendations.")

                # Visualization
                st.bar_chart(feature_importance_df.set_index("Feature"))

                # Encouraging message
                st.markdown("""
                    Explore each career path to discover how your unique strengths can help you thrive. 
                    Remember, career choices are not set in stone—they're stepping stones to your growth!
                """)

            except KeyError as e:
                st.error(f"Feature mismatch: {e}. Check feature alignment.")
            except Exception as e:
                st.error(f"Unexpected error: {e}. Verify your setup.")
        else:
            st.warning("Complete the questionnaire to view detailed insights.")
