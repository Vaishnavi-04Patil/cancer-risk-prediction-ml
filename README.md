# Cancer Risk Prediction Using Machine Learning

## Project Overview

This project develops a Machine Learning model to predict an individual's cancer risk level as **Low, Medium, or High** based on demographic, lifestyle, environmental, genetic, and medical features.

The project focuses not only on overall model accuracy but also on handling **class imbalance** and improving the prediction of the minority **High-risk** class.

Multiple machine learning experiments were conducted using **Random Forest and XGBoost**, along with **SMOTE, class weighting, and Optuna hyperparameter optimization**.

The final selected model was an **Optuna-tuned, class-weighted XGBoost model**.

---

## Problem Statement

Develop a machine learning model that can accurately predict the risk level of an individual as **High, Medium, or Low** using a combination of demographic, behavioral, and health-related features.

The objective is to achieve good overall predictive performance while handling class imbalance effectively, particularly ensuring that the minority High-risk class is identified with meaningful recall and F1-score.

---

## Business Understanding

Machine learning can help identify individuals who may have higher cancer-risk profiles based on demographic, lifestyle, environmental, genetic, and medical factors.

Such predictive analysis can support:

* Early risk identification
* Preventive healthcare strategies
* Risk segmentation
* Data-driven healthcare analysis
* Further investigation of high-risk individuals

> **Disclaimer:** This project is created for educational and demonstration purposes only. It is not intended for clinical diagnosis, medical decision-making, or patient treatment.

---

## Dataset & Feature Summary

The dataset contains demographic, lifestyle, environmental, genetic, and medical features.

### Demographic Features

* **Age**
* **Gender** — 0 = Female, 1 = Male
* **BMI**

### Lifestyle & Environmental Features

These features are represented using a **0–10 index**:

* Smoking
* Alcohol_Use
* Obesity
* Diet_Red_Meat
* Diet_Salted_Processed
* Fruit_Veg_Intake
* Physical_Activity
* Physical_Activity_Level
* Air_Pollution
* Occupational_Hazards
* Calcium_Intake

### Genetic & Medical Features

Binary 0/1 indicators:

* Family_History
* BRCA_Mutation
* H_Pylori_Infection

### Engineered Features

* Overall_Risk_Score
* Risk_Level

### Data Notes

* Prostate cancer occurs only when Gender = 1.
* A small number of male cases appear in Breast cancer.
* The Risk_Level target is moderately imbalanced, with Medium risk being the majority class.
* The composite risk score generally aligns with exposure intensity, increasing with factors such as smoking and air pollution and decreasing with higher fruit and vegetable intake.

---

# Exploratory Data Analysis

## 1. Distribution of Risk Levels

The dataset contains three risk categories:

* Low
* Medium
* High

The **Medium-risk class is the majority**, followed by Low risk, while High risk represents the minority class.

This class imbalance became an important consideration during model development.

---

## 2. Cancer Type Distribution by Gender

EDA was performed to understand the relationship between gender and cancer type.

### Female Patients

Breast cancer was the most common cancer type among females, with **455 female patients**.

Other cancer types observed among females included:

* Colon
* Lung
* Skin

### Male Patients

Prostate cancer was the most common cancer type among males, with **305 patients**.

A small number of male patients were also associated with Breast cancer, with **5 cases**.

Other cancer types among males included:

* Colon
* Lung
* Skin

---

## 3. Cancer Types Among Patients Below 30

Patients younger than 30 years were analyzed separately.

There were only **3 patients below the age of 30**, and all were females with Lung cancer.

---

## 4. Factors Associated With Higher Risk

EDA indicated that several factors were associated with higher risk levels, including:

* Air Pollution
* Smoking
* Alcohol Use
* Salted/Processed Food Intake
* Occupational Hazards
* Red Meat Intake
* Obesity

These variables showed higher exposure levels among higher-risk groups.

---

## 5. BMI and Risk Level

The relationship between BMI and Risk Level was also investigated.

The analysis indicated that **BMI did not have a strong direct relationship with Risk Level** in this dataset.

---

# Data Preprocessing

Before building the final models, the dataset was examined for potential issues that could affect model performance.

One of the most important findings was **data leakage**.

The feature `Overall_Risk_Score` was initially included as an input feature. Since this composite feature was directly related to the target `Risk_Level`, the model could use it as a shortcut instead of learning meaningful patterns from the underlying health and lifestyle variables.

Therefore, `Overall_Risk_Score` was removed from the model input features.

This resulted in a more realistic evaluation of the machine learning models.

---

# Machine Learning Approach

Several machine learning experiments were performed to identify a suitable model.

The main algorithms used were:

* Logistic Regression
* Random Forest
* XGBoost

Additional techniques were applied to address class imbalance and improve model performance:

* SMOTE
* Class Weighting
* Optuna Hyperparameter Optimization

---

# Model Experiments

## Experiment 1 — Initial Logistic Regression & Random Forest

Initial Logistic Regression and Random Forest models produced very high overall performance.

Further investigation revealed that the `Overall_Risk_Score` feature was causing **data leakage**.

The feature was removed before continuing with model development.

---

## Experiment 2 — Random Forest

After removing the leakage feature, Random Forest achieved approximately:

**Accuracy: 84%**

However, the model struggled with the High-risk class and correctly identified only **7 out of 20 High-risk patients**.

This demonstrated the importance of evaluating class-level performance instead of relying only on accuracy.

---

## Experiment 3 — Optuna Tuned Random Forest

Optuna was used to search for better Random Forest hyperparameters.

The model achieved approximately:

**Accuracy: 83%**

High-risk prediction improved only slightly, with approximately **5 out of 20 High-risk patients** correctly identified in the reported experiment.

---

## Experiment 4 — SMOTE + Optuna + Random Forest

SMOTE was introduced to address class imbalance by generating synthetic samples for minority classes.

The resulting model achieved approximately:

**Accuracy: 83%**

High-risk performance improved, with **9 out of 20 High-risk patients** correctly predicted.

---

## Experiment 5 — High-Risk Focused Random Forest + SMOTE

The next experiment focused more heavily on improving High-risk predictions using SMOTE and Random Forest.

Results:

* **Accuracy: 80%**
* **12 out of 20 High-risk patients** correctly predicted

This experiment demonstrated that improving minority-class recall can involve a trade-off with overall accuracy.

---

## Experiment 6 — XGBoost + SMOTE

XGBoost was then evaluated with SMOTE.

Results:

* **Accuracy: 85%**
* High-risk recall remained limited compared with the desired objective.

XGBoost provided stronger overall performance than the Random Forest baseline.

---

## Experiment 7 — Optuna + XGBoost With High-Risk Recall Optimization

Optuna was used to optimize XGBoost specifically toward improving High-risk recall.

The model successfully improved High-risk identification:

* **13 out of 20 High-risk patients** correctly predicted
* Overall accuracy decreased to approximately **61%**

This highlighted the trade-off between maximizing minority-class recall and maintaining overall model performance.

---

## Experiment 8 — Class-Weighted XGBoost

Instead of using SMOTE, class weighting was introduced to give greater importance to minority classes.

Results:

* **Accuracy: 64%**
* **15 out of 20 High-risk patients** correctly predicted

This produced substantially stronger High-risk detection, although overall accuracy decreased.

---

## Experiment 9 — Optuna Tuned Class-Weighted XGBoost

The final experiment combined:

* XGBoost
* Class weighting
* Optuna hyperparameter optimization

The objective was to achieve a better balance between overall performance and minority-class prediction.

The model achieved:

**Accuracy: 88%**

Based on the overall evaluation, this model was selected as the final model for deployment.

---

| Model                                   | Accuracy | Macro F1 | Weighted F1 | High Recall | Low Recall | Medium Recall |
| --------------------------------------- | -------: | -------: | ----------: | ----------: | ---------: | ------------: |
| Random Forest — Balanced, No Leakage    |     0.84 |     0.64 |        0.83 |        0.35 |       0.58 |          0.92 |
| Optuna Tuned RF — Macro F1 Focus        |     0.83 |     0.65 |        0.83 |        0.45 |       0.63 |          0.89 |
| Optuna Tuned RF — High Recall Focus     |     0.80 |     0.65 |        0.81 |        0.60 |       0.74 |          0.82 |
| Baseline XGBoost — SMOTE                |     0.85 |     0.68 |        0.85 |        0.45 |       0.68 |          0.92 |
| Class-Weighted XGBoost                  |     0.64 |     0.54 |        0.67 |    **0.75** |   **0.82** |          0.59 |
| **Optuna Tuned Class-Weighted XGBoost** | **0.88** | **0.72** |    **0.87** |    **0.45** |   **0.78** |      **0.92** |


---

# Final Model

The final selected model is:

### Optuna Tuned Class-Weighted XGBoost

The model was selected because it provided the best overall balance between:

* Accuracy
* Macro F1
* Weighted F1
* High-risk class performance
* Low-risk class performance
* Medium-risk class performance

### Final Performance

* **Accuracy:** 88%
* **Macro F1:** 0.72
* **Weighted F1:** 0.87
* **High-risk Recall:** 0.45
* **Low-risk Recall:** 0.78
* **Medium-risk Recall:** 0.92

The model was subsequently saved as a `.pkl` file and integrated into a Streamlit application.

---

# Why SMOTE Was Used

The Risk_Level target contains an imbalance, with Medium risk representing the majority class and High risk representing a minority class.

**SMOTE (Synthetic Minority Over-sampling Technique)** was used during several experiments to generate synthetic samples for minority classes.

This helps prevent a machine learning model from becoming overly biased toward the majority class.

However, SMOTE was not used in the final selected model. The final approach used **class weighting with XGBoost**, combined with Optuna hyperparameter optimization.

---

# Key Learning: Data Leakage

One of the most important findings during this project was the impact of **data leakage**.

Initially, `Overall_Risk_Score` was included among the model features.

Because this score was directly related to the target `Risk_Level`, the model could rely on this engineered feature rather than learning the underlying relationships between the health-related variables and risk.

After identifying the leakage, the feature was removed from the model inputs.

This produced a more realistic evaluation and highlighted the importance of understanding the origin and relationship of features before training a machine learning model.

---

# Streamlit Application

A Streamlit application was developed to provide an interactive interface for the cancer risk prediction model.

The application allows users to provide relevant health and lifestyle inputs and obtain a predicted risk category.

The trained XGBoost model is stored as:

```text
model_xgb_new.pkl
```

The Streamlit application is located in:

```text
app/app.py
```

---

# Project Structure

```text
cancer-risk-prediction-ml/
│
├── EDA_cancer.ipynb
├── Testing_Models.ipynb
├── README.md
├── app/
│   └── app.py
├── model_xgb_new.pkl
└── requirements.txt
```

---

# Technologies Used

### Programming & Data Analysis

* Python
* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Random Forest
* XGBoost
* Logistic Regression
* SMOTE

### Hyperparameter Optimization

* Optuna

### Deployment

* Streamlit

### Development Environment

* Jupyter Notebook
* GitHub

---

# How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/Vaishnavi-04Patil/cancer-risk-prediction-ml.git
```

## 2. Navigate to the Project Directory

```bash
cd cancer-risk-prediction-ml
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Streamlit Application

```bash
streamlit run app/app.py
```

The application will then open in your browser.

---

# Conclusion

This project demonstrates an end-to-end machine learning workflow for predicting cancer risk levels.

The project covered:

* Exploratory Data Analysis
* Feature analysis
* Data preprocessing
* Data leakage detection
* Class imbalance analysis
* Logistic Regression
* Random Forest
* XGBoost
* SMOTE
* Class weighting
* Optuna hyperparameter optimization
* Model evaluation
* Streamlit deployment

The final **Optuna-tuned Class-Weighted XGBoost model** achieved **88% accuracy** and was selected as the final model based on its overall performance and balance across the three risk classes.

A key learning from the project was that **model evaluation should go beyond accuracy**, particularly when working with imbalanced datasets. Evaluating class-specific recall, F1-score, and the impact of data leakage was essential to selecting a more reliable model.

---

## Disclaimer

This project is intended for **educational and portfolio demonstration purposes only**. The predictions generated by this application should not be used for medical diagnosis, clinical decisions, or treatment recommendations.
