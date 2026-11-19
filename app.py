# ==============================================================================
# Project: Breast Cancer Classification Using Machine Learning
# File: app.py
# Description: Clean and simple Streamlit web application for college project
#              demonstration. Uses the model trained on data/data.csv.
# ==============================================================================

import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Classification",
    layout="wide"
)

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS TO LOAD DATA AND SAVED MODEL
# ------------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    """Loads the dataset from data/data.csv."""
    path = os.path.join("data", "data.csv")
    if not os.path.exists(path):
        st.error(f"Dataset file '{path}' not found.")
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_trained_model():
    """Loads the trained model artifacts saved by train_model.py."""
    path = "model.joblib"
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ------------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------------------
st.sidebar.title("Navigation")
menu_choice = st.sidebar.radio(
    "Go to Section:",
    [
        "1. Project Overview",
        "2. Dataset Information",
        "3. Exploratory Data Analysis",
        "4. Model Comparison & Evaluation",
        "5. Cancer Prediction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project:** Breast Cancer Classification\n\n"
    "**Student:** College ML Project\n\n"
    "**Dataset:** Wisconsin Diagnostic Dataset"
)


# ------------------------------------------------------------------------------
# SECTION 1: PROJECT OVERVIEW
# ------------------------------------------------------------------------------
if menu_choice == "1. Project Overview":
    st.title("Breast Cancer Classification Using Machine Learning")

    st.write(
        "Welcome to the Breast Cancer Classification project. This machine learning application "
        "predicts whether a breast mass tumor is **Benign (non-cancerous)** or **Malignant (cancerous)** "
        "based on characteristics computed from digitized images of fine needle aspirates (FNA) of breast masses."
    )

    st.subheader("Project Objectives")
    st.write(
        "- Perform exploratory data analysis (EDA) to understand feature distributions and correlations.\n"
        "- Clean and preprocess the Wisconsin Breast Cancer dataset.\n"
        "- Train and compare three standard machine learning classifiers:\n"
        "  1. **Gaussian Naive Bayes** (probabilistic baseline studied in class)\n"
        "  2. **Logistic Regression** (linear classification model)\n"
        "  3. **Support Vector Machine (SVM)** (boundary-margin optimization)\n"
        "- Evaluate models using Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.\n"
        "- Provide an interactive prediction tool for demonstration."
    )

    st.warning(
        "⚠️ **Disclaimer:** This project is developed exclusively for academic and educational purposes "
        "as part of a college machine learning course. It is NOT intended or certified for actual clinical "
        "diagnosis or real-world medical decision-making."
    )


# ------------------------------------------------------------------------------
# SECTION 2: DATASET INFORMATION
# ------------------------------------------------------------------------------
elif menu_choice == "2. Dataset Information":
    st.title("Dataset Information")

    df = load_dataset()
    if df is not None:
        st.write(
            f"The dataset is loaded from `data/data.csv`. It contains **{df.shape[0]} rows** "
            f"and **{df.shape[1]} original columns**."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", df.shape[0])
        with col2:
            st.metric("Benign Samples (B)", int((df["diagnosis"] == "B").sum()))
        with col3:
            st.metric("Malignant Samples (M)", int((df["diagnosis"] == "M").sum()))

        st.subheader("First 5 Rows of the Dataset")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

        st.subheader("Data Cleaning Summary")
        st.write(
            "- **`id` Column:** Dropped because patient IDs are arbitrary unique numbers and not predictive biological features.\n"
            "- **`Unnamed: 32` Column:** Dropped because it contains 100% missing (NaN) values due to trailing commas in the CSV.\n"
            "- **Duplicate Rows:** 0 duplicate rows found in the dataset.\n"
            "- **Target Variable (`diagnosis`):** 'B' (Benign, 62.7%) and 'M' (Malignant, 37.3%)."
        )


# ------------------------------------------------------------------------------
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------------------------
elif menu_choice == "3. Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    st.write("Visualizations analyzing the distribution, relationships, and correlations of the features.")

    plot_option = st.selectbox(
        "Choose an EDA Visualization to Inspect:",
        [
            "Target Class Distribution",
            "Correlation Heatmap (Mean Features)",
            "Feature Histograms",
            "Feature Box Plots by Diagnosis",
            "Scatter Plot: Radius vs Texture"
        ]
    )

    if plot_option == "Target Class Distribution":
        st.subheader("Distribution of Diagnosis")
        if os.path.exists("plots/target_distribution.png"):
            st.image("plots/target_distribution.png", use_container_width=True)
        st.write(
            "**Observation:** The dataset contains 357 Benign (B) cases and 212 Malignant (M) cases. "
            "Because the classes are somewhat imbalanced (63% Benign, 37% Malignant), we use stratified splitting "
            "(`stratify=y`) during train-test split to maintain this exact ratio in both sets."
        )

    elif plot_option == "Correlation Heatmap (Mean Features)":
        st.subheader("Correlation Heatmap")
        if os.path.exists("plots/correlation_heatmap.png"):
            st.image("plots/correlation_heatmap.png", use_container_width=True)
        st.write(
            "**Observation:** Geometric features such as `radius_mean`, `perimeter_mean`, and `area_mean` "
            "show extremely high correlation (close to 1.0). This is expected because perimeter and area "
            "are mathematical functions of radius. This feature correlation violates the conditional independence "
            "assumption of Naive Bayes, which explains why Naive Bayes has a slightly lower recall."
        )

    elif plot_option == "Feature Histograms":
        st.subheader("Feature Distributions (Histograms)")
        if os.path.exists("plots/feature_histograms.png"):
            st.image("plots/feature_histograms.png", use_container_width=True)
        st.write(
            "**Observation:** For features like `radius_mean`, `perimeter_mean`, and `area_mean`, Malignant tumors "
            "(red) generally display significantly larger values than Benign tumors (green), making these strong "
            "separators for classification algorithms."
        )

    elif plot_option == "Feature Box Plots by Diagnosis":
        st.subheader("Box Plots Grouped by Diagnosis")
        if os.path.exists("plots/feature_boxplots.png"):
            st.image("plots/feature_boxplots.png", use_container_width=True)
        st.write(
            "**Observation:** The box plots clearly illustrate the differences in medians and quartiles. "
            "Malignant cases consistently show higher values and wider spread across all four selected features."
        )

    elif plot_option == "Scatter Plot: Radius vs Texture":
        st.subheader("Scatter Plot: Radius Mean vs Texture Mean")
        if os.path.exists("plots/scatter_plot.png"):
            st.image("plots/scatter_plot.png", use_container_width=True)
        st.write(
            "**Observation:** Plotting `radius_mean` against `texture_mean` shows a clear separation boundary "
            "between Benign and Malignant clusters, explaining why linear classifiers and SVM achieve high accuracy."
        )


# ------------------------------------------------------------------------------
# SECTION 4: MODEL COMPARISON & EVALUATION
# ------------------------------------------------------------------------------
elif menu_choice == "4. Model Comparison & Evaluation":
    st.title("Model Training & Evaluation Results")

    artifacts = load_trained_model()
    if artifacts is None:
        st.error("Model artifacts not found. Please run `py train_model.py` first.")
    else:
        st.subheader("Performance Comparison Table")
        comparison_df = artifacts["comparison_table"]
        st.dataframe(comparison_df.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}"
        }), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Accuracy Comparison")
            if os.path.exists("plots/model_comparison.png"):
                st.image("plots/model_comparison.png", use_container_width=True)

        with col2:
            st.subheader("Confusion Matrices")
            if os.path.exists("plots/confusion_matrices.png"):
                st.image("plots/confusion_matrices.png", use_container_width=True)

        st.subheader("Selected Model Explanation")
        st.write(
            f"**Selected Best Model:** **{artifacts['model_name']}**\n\n"
            f"- It achieved the highest test accuracy and the best F1-Score among all tested algorithms.\n"
            f"- It achieved high Recall on malignant tumors (identifying 39 out of 42 malignant cases in the test set).\n"
            f"- Gaussian Naive Bayes was trained as a baseline without scaling; however, due to correlated features, "
            f"it produced 7 false negatives.\n"
            f"- Logistic Regression and Support Vector Machine benefited significantly from `StandardScaler` normalization."
        )

        st.subheader("Why False Negatives Matter in Breast Cancer Detection")
        st.write(
            "In medical diagnostic systems, errors are not equal:\n\n"
            "- **False Positive (FP):** A benign tumor is predicted as malignant. The patient undergoes additional testing (e.g. biopsy), causing temporary anxiety, but no life is immediately lost.\n"
            "- **False Negative (FN):** A malignant cancer is predicted as benign. The patient is falsely reassured and cancer remains untreated, which can lead to disease progression and severe health consequences.\n\n"
            "Therefore, **high Recall (sensitivity)** for Malignant cases is the primary metric of concern."
        )


# ------------------------------------------------------------------------------
# SECTION 5: CANCER PREDICTION
# ------------------------------------------------------------------------------
elif menu_choice == "5. Cancer Prediction":
    st.title("Cancer Diagnosis Prediction")
    st.write("Enter the 30 cell nucleus measurements or load sample cases to test the trained model.")

    artifacts = load_trained_model()
    if artifacts is None:
        st.error("Model file 'model.joblib' not found. Please run `py train_model.py` first.")
    else:
        feature_names = artifacts["feature_names"]
        model = artifacts["model"]
        scaler = artifacts["scaler"]
        model_name = artifacts["model_name"]

        st.info(f"Loaded Trained Model: **{model_name}**")

        # Preset test samples taken directly from data/data.csv
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

        preset = st.selectbox(
            "Load Sample Values for Quick Viva Demonstration:",
            ["Custom Input", "Sample 1: Malignant Case (Actual Dataset Row 0)", "Sample 2: Benign Case (Actual Dataset Row 19)"]
        )

        current_values = {}
        if preset == "Sample 1: Malignant Case (Actual Dataset Row 0)":
            current_values = sample_malignant
        elif preset == "Sample 2: Benign Case (Actual Dataset Row 19)":
            current_values = sample_benign

        st.subheader("Input Tumor Features")

        # Organize 30 features into 3 tabs
        tab1, tab2, tab3 = st.tabs(["Mean Features", "Standard Error (SE) Features", "Worst (Largest) Features"])

        user_inputs = {}

        with tab1:
            st.write("Average cell nucleus measurements:")
            mean_cols = [c for c in feature_names if "_mean" in c]
            cols = st.columns(2)
            for i, col in enumerate(mean_cols):
                default_val = float(current_values.get(col, 15.0 if "area" in col or "perimeter" in col else 1.0))
                with cols[i % 2]:
                    user_inputs[col] = st.number_input(
                        f"{col}",
                        value=default_val,
                        format="%.5f",
                        key=f"input_{col}"
                    )

        with tab2:
            st.write("Standard error cell nucleus measurements:")
            se_cols = [c for c in feature_names if "_se" in c]
            cols = st.columns(2)
            for i, col in enumerate(se_cols):
                default_val = float(current_values.get(col, 0.5 if "area" in col or "perimeter" in col else 0.05))
                with cols[i % 2]:
                    user_inputs[col] = st.number_input(
                        f"{col}",
                        value=default_val,
                        format="%.5f",
                        key=f"input_{col}"
                    )

        with tab3:
            st.write("Largest/worst cell nucleus measurements:")
            worst_cols = [c for c in feature_names if "_worst" in c]
            cols = st.columns(2)
            for i, col in enumerate(worst_cols):
                default_val = float(current_values.get(col, 20.0 if "area" in col or "perimeter" in col else 1.5))
                with cols[i % 2]:
                    user_inputs[col] = st.number_input(
                        f"{col}",
                        value=default_val,
                        format="%.5f",
                        key=f"input_{col}"
                    )

        st.markdown("---")
        if st.button("Predict Tumor Diagnosis"):
            input_df = pd.DataFrame([user_inputs])[feature_names]

            # Scale if model is Logistic Regression or Support Vector Machine
            if model_name in ["Logistic Regression", "Support Vector Machine"]:
                input_scaled = scaler.transform(input_df)
            else:
                input_scaled = input_df

            prediction = model.predict(input_scaled)[0]

            st.subheader("Prediction Result:")
            if prediction == "M":
                st.error("🚨 **Prediction: Malignant (Cancerous)**\n\nThe features correspond to a malignant breast tumor.")
            else:
                st.success("✅ **Prediction: Benign (Non-Cancerous)**\n\nThe features correspond to a benign breast mass.")

            st.info(
                f"Prediction produced by **{model_name}** using standardized features from `data/data.csv`."
            )

            st.warning(
                "⚠️ **Educational Project Notice:** This prediction is generated by an academic machine learning "
                "model and is strictly for educational demonstration. It must not be used for medical decisions."
            )
