# From Prediction to Action: Personalized Explainable AI Profiles for At-Risk Students

# Project Overview
This project addresses the "black-box" nature of high-accuracy predictive models in education. By leveraging the Open University Learning Analytics Dataset (OULAD), I developed a multi-layered framework that not only predicts student dropout with high precision but also provides human-understandable justifications for these predictions.
+2

The core innovation is the translation of complex model outputs into actionable student personas and the empirical testing of explanation reliability through robustness analysis.

# Tech Stack & Tools

Languages: Python.

Machine Learning: XGBoost, Logistic Regression, Random Forest, Multi-Layer Perceptron (MLP).

Explainable AI (XAI): SHAP (SHapley Additive exPlanations), LIME (Local Interpretable Model-agnostic Explanations).

Data Engineering: Pandas, NumPy, Scikit-learn (One-Hot Encoding, SMOTE).

Deployment & Visualization: Streamlit (Dashboard), Matplotlib, Seaborn.

Environment: Google Colab, Google Cloud Dataproc (for Spark optimization components).


# Key Features & Methodology

Multi-Model Benchmarking: Compared high-performance "black-box" models (XGBoost) against transparent "white-box" benchmarks (Logistic Regression) to evaluate the trade-off between accuracy and interpretability.

Temporal Explainability: Conducted a week-by-week analysis over a 26-week semester. Discovered that predictive factors shift from demographic proxies in early weeks to academic performance indicators (e.g., avg_score) in later stages.

Fairness & Bias Auditing: Performed a systematic audit across sensitive subgroups (Gender, Age, Disability) using SHAP and LIME to detect localized biases that global metrics often mask.

Robustness Testing: Subjected model explanations to data perturbations (e.g., a 5% increase in total_clicks) to test stability. Results confirmed SHAP's reliability over LIME for high-stakes decision-making.

Actionable Student Personas: Applied K-Means clustering to SHAP values to segment students into distinct personas, such as "Evaluation-Driven" or "Behavioral/Demographic-Driven," allowing for tailored interventions.

# Results Summary

Model Performance: XGBoost achieved a peak ROC-AUC of 0.9628 and an F1-score of 0.90.

Stability: SHAP feature rankings remained consistent under perturbation, whereas LIME showed significant volatility, completely removing top features after minor data shifts.

Impact: Identified num_submitted_assessments and avg_score as the most critical factors for identifying dropout risk across all models.

# Project Structure

01_Data_Preprocessing.ipynb: Data integration of 7 OULAD CSV files, cleaning, and SMOTE implementation.
02_Baseline_Models.ipynb: Training and cross-validation of initial classifiers.
03_Hyperparameter_Tuning.ipynb: Randomized and Grid Search optimization.
04_XAI_Implementation.ipynb: Core global and local explanation logic using SHAP and LIME.
05_Advanced_Analyses.ipynb: Temporal, Fairness, Robustness, and Clustering analyses.

# References
Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics Dataset (OULAD).

Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions.

Ribeiro, M. T., et al. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier.
