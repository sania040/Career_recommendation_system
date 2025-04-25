import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import shap

# Ensemble model: Random Forest for recommendations
def train_ensemble_model(data):
    """
    Trains a Random Forest Classifier using the career dataset.
    Returns the trained model and features.
    """
    features = [
        "O_score", "C_score", "E_score", "A_score", "N_score",
        "Numerical Aptitude", "Spatial Aptitude",
        "Perceptual Aptitude", "Abstract Reasoning", "Verbal Reasoning"
    ]
    target = "Career"
    X, y = data[features], data[target]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    model.fit(X, y)
    return model, features

def calculate_overall_score(personality_responses, aptitude_responses):
    """
    Convert questionnaire responses into overall scores for the user.
    """
    personality_map = {"Strongly Disagree": 1, "Disagree": 3, "Neutral": 5, "Agree": 7, "Strongly Agree": 10}
    aptitude_map = {"Very Poor": 1, "Poor": 3, "Average": 5, "Good": 7, "Excellent": 10}
    
    personality_scores = [personality_map[response] for response in personality_responses]
    aptitude_scores = [aptitude_map[response] for response in aptitude_responses]
    
    return {
        "O_score": personality_scores[0],
        "C_score": personality_scores[1],
        "E_score": personality_scores[2],
        "A_score": personality_scores[3],
        "N_score": personality_scores[4],
        "Numerical Aptitude": aptitude_scores[0],
        "Spatial Aptitude": aptitude_scores[1],
        "Perceptual Aptitude": aptitude_scores[2],
        "Abstract Reasoning": aptitude_scores[3],
        "Verbal Reasoning": aptitude_scores[4],
    }

def recommend_career_with_rf(user_scores, model, data, features):
    """
    Recommend careers using the trained Random Forest model.
    """
    user_input = pd.DataFrame([user_scores])
    probabilities = model.predict_proba(user_input[features])[0]
    top_indices = np.argsort(probabilities)[-5:][::-1]
    recommendations = model.classes_[top_indices]
    scores = probabilities[top_indices]
    return list(zip(recommendations, scores))

def explain_recommendations(model, user_scores, features):
    """
    Explain career recommendations using SHAP.
    """
    explainer = shap.TreeExplainer(model.named_steps["rf"])
    user_input = pd.DataFrame([user_scores])
    shap_values = explainer.shap_values(user_input[features])
    return explainer, shap_values
