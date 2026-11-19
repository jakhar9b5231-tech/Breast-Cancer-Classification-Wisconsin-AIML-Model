# ==============================================================================
# Project: Breast Cancer Classification Using Machine Learning
# File: predict.py
# Description: Simple prediction program that loads the saved model, validates
#              new input features against data.csv columns, and predicts the diagnosis.
# ==============================================================================

import os
import joblib
import pandas as pd

# Path to the trained model file
MODEL_FILE = "model.joblib"


def load_saved_model(filepath=MODEL_FILE):
    """Loads the trained model, scaler, and feature names saved by train_model.py."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Model file '{filepath}' not found. Please run 'train_model.py' first."
        )
    return joblib.load(filepath)


def predict_tumor(input_dict):
    """
    Validates input features, scales them if necessary, and predicts the tumor class.
    
    Parameters:
        input_dict (dict): Dictionary with feature names as keys and measurements as values.
    
    Returns:
        prediction (str): 'B' for Benign or 'M' for Malignant.
        label (str): Full readable text label.
    """
    artifacts = load_saved_model()
    model = artifacts["model"]
    model_name = artifacts["model_name"]
    scaler = artifacts["scaler"]
    expected_features = artifacts["feature_names"]

    # 1. Validate that all required features are present
    missing_features = [f for f in expected_features if f not in input_dict]
    if missing_features:
        raise ValueError(
            f"Input is missing required features: {missing_features}"
        )

    # 2. Arrange features in the exact column order expected by the model
    input_df = pd.DataFrame([input_dict])[expected_features]

    # 3. Apply StandardScaler if model is Logistic Regression or Support Vector Machine
    if model_name in ["Logistic Regression", "Support Vector Machine"]:
        input_processed = scaler.transform(input_df)
    else:
        # Gaussian Naive Bayes was trained on unscaled features
        input_processed = input_df

    # 4. Predict diagnosis
    prediction = model.predict(input_processed)[0]
    
    label = "Malignant (Cancerous)" if prediction == "M" else "Benign (Non-Cancerous)"
    return prediction, label, model_name


if __name__ == "__main__":
    print("=" * 60)
    print("BREAST CANCER CLASSIFICATION - PREDICTION PROGRAM")
    print("=" * 60)

    # Example sample taken directly from a Malignant case in data/data.csv (Row 0)
    sample_malignant = {
        'radius_mean': 17.99, 'texture_mean': 10.38, 'perimeter_mean': 122.8,
        'area_mean': 1001.0, 'smoothness_mean': 0.1184, 'compactness_mean': 0.2776,
        'concavity_mean': 0.3001, 'concave points_mean': 0.1471, 'symmetry_mean': 0.2419,
        'fractal_dimension_mean': 0.07871, 'radius_se': 1.095, 'texture_se': 0.9053,
        'perimeter_se': 8.589, 'area_se': 153.4, 'smoothness_se': 0.006399,
        'compactness_se': 0.04904, 'concavity_se': 0.05373, 'concave points_se': 0.01587,
        'symmetry_se': 0.03003, 'fractal_dimension_se': 0.006193, 'radius_worst': 25.38,
        'texture_worst': 17.33, 'perimeter_worst': 184.6, 'area_worst': 2019.0,
        'smoothness_worst': 0.1622, 'compactness_worst': 0.6656, 'concavity_worst': 0.7119,
        'concave points_worst': 0.2654, 'symmetry_worst': 0.4601,
        'fractal_dimension_worst': 0.1189
    }

    # Example sample taken directly from a Benign case in data/data.csv (Row 19)
    sample_benign = {
        'radius_mean': 13.54, 'texture_mean': 14.36, 'perimeter_mean': 87.46,
        'area_mean': 566.3, 'smoothness_mean': 0.09779, 'compactness_mean': 0.08129,
        'concavity_mean': 0.06664, 'concave points_mean': 0.04781, 'symmetry_mean': 0.1885,
        'fractal_dimension_mean': 0.05766, 'radius_se': 0.2699, 'texture_se': 0.7886,
        'perimeter_se': 2.058, 'area_se': 23.56, 'smoothness_se': 0.008462,
        'compactness_se': 0.0146, 'concavity_se': 0.02387, 'concave points_se': 0.01315,
        'symmetry_se': 0.0198, 'fractal_dimension_se': 0.0023, 'radius_worst': 15.11,
        'texture_worst': 19.26, 'perimeter_worst': 99.7, 'area_worst': 711.2,
        'smoothness_worst': 0.144, 'compactness_worst': 0.1773, 'concavity_worst': 0.239,
        'concave points_worst': 0.1288, 'symmetry_worst': 0.2977,
        'fractal_dimension_worst': 0.07259
    }

    print("\n--- Testing Sample 1 (Expected: Malignant) ---")
    pred1, label1, model_used = predict_tumor(sample_malignant)
    print(f"Model Used: {model_used}")
    print(f"Raw Prediction: {pred1}")
    print(f"Result: {label1}")

    print("\n--- Testing Sample 2 (Expected: Benign) ---")
    pred2, label2, model_used = predict_tumor(sample_benign)
    print(f"Model Used: {model_used}")
    print(f"Raw Prediction: {pred2}")
    print(f"Result: {label2}")

    print("\n" + "=" * 60)
    print("DISCLAIMER: This project is for educational and study purposes only.")
    print("It is not a substitute for professional clinical medical diagnosis.")
    print("=" * 60)
