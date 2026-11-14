# ✈️ Flight Delay Prediction Using Machine Learning

## Flight Delay Prediction: A Comparative Machine Learning Approach (Logistic Regression vs. Random Forest)

A comparative study of supervised machine learning algorithms for binary flight delay classification, evaluating Logistic Regression and Random Forest using aviation-related operational, scheduling, route, and environmental attributes.

The implementation integrates heterogeneous data preprocessing, stratified model validation, classification performance analysis, and five-fold cross-validated hyperparameter optimization using Scikit-learn. Particular emphasis is placed on evaluating minority-class predictive performance under class imbalance, analysing precision–recall trade-offs, and employing F1-score as the primary optimization criterion for hyperparameter tuning and comparative model selection.

**Presented by:** Vedant Kumawat

**Programme:** MBA – Business Analytics, Parul University

---

## 1. Project Overview

Flight delays affect passengers, airline operations, scheduling and customer satisfaction.

This project uses machine learning classification algorithms to predict whether a flight will be delayed.

**Business Question:**

Can we predict whether a flight will be delayed using information available before departure?

### Project Objectives

- Understand and clean the aviation dataset.
- Perform Exploratory Data Analysis (EDA).
- Identify important features.
- Build preprocessing pipelines.
- Train Logistic Regression and Random Forest models.
- Evaluate model performance.
- Perform hyperparameter tuning using GridSearchCV.
- Compare baseline and tuned models.
- Select the best-performing model.

## 2. Dataset Information

| Description | Value |
|---|---:|
| Total records | 10,000 |
| Original columns | 25 |
| Selected features | 16 |
| Training data | 80% |
| Testing data | 20% |
| Target variable | Flight_Delayed |

### Target Variable

- 0 = Not Delayed
- 1 = Delayed

### Target Distribution

| Class | Records | Percentage |
|---|---:|---:|
| Not Delayed | 7,395 | 73.95% |
| Delayed | 2,605 | 26.05% |

**Note:** This is a synthetic educational dataset, not actual airline operational data.

## 3. Machine Learning Workflow

1. Import libraries
2. Load dataset
3. Data understanding
4. Data quality checks
5. Data cleaning
6. Exploratory Data Analysis
7. Correlation analysis
8. Feature selection
9. Train-test split
10. Data preprocessing
11. Logistic Regression
12. Random Forest
13. Model evaluation
14. Confusion matrix
15. Hyperparameter tuning
16. Final model comparison

## 4. Algorithms Used

### Logistic Regression

A supervised classification algorithm that estimates the probability of a flight being delayed.

### Random Forest

An ensemble learning algorithm that combines multiple decision trees to make predictions.

## 5. Data Preprocessing

**Numerical features:**
- Missing value imputation using median.
- StandardScaler for Logistic Regression.

**Categorical features:**
- Missing value imputation using most frequent value.
- One-Hot Encoding.

Preprocessing is implemented using Scikit-learn Pipeline and ColumnTransformer.

## 6. Model Evaluation

The following metrics are used:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## 7. Baseline Model Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 76.1% | 69.4% | 14.8% | 24.4% |
| Random Forest | 74.7% | 59.0% | 9.4% | 16.2% |

## 8. Hyperparameter Tuning

GridSearchCV is used with five-fold cross-validation.

**Logistic Regression:**
- C: 0.01, 0.1, 1, 10
- class_weight: None, balanced

**Random Forest:**
- n_estimators: 100, 200
- max_depth: None, 10, 20
- min_samples_split: 2, 5
- class_weight: None, balanced

Scoring metric: F1-score.

## 9. Final Model Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Tuned Logistic Regression | 65.6% | 39.5% | 60.1% | 47.6% |
| Tuned Random Forest | 71.3% | 44.3% | 39.7% | 41.9% |

## 10. Final Conclusion

Tuned Logistic Regression is selected as the preferred model because it achieves:

- Higher recall: 60.1%
- Higher F1-score: 47.6%

Random Forest achieves higher accuracy and precision.

Logistic Regression is preferable when identifying more actual delayed flights is the main priority.

## 11. Business Applications

- Airline operations and planning.
- Flight scheduling.
- Risk identification.
- Proactive passenger communication.
- Operational decision support.

## 12. Limitations

- **Synthetic Dataset:** The model was developed using synthetic aviation data; therefore, its predictive performance may not generalize to actual airline operations.

- **Class Imbalance:** The dataset contains an unequal distribution of delayed and non-delayed flights, which can influence classification performance and minority-class detection.

- **False Predictions:** Despite hyperparameter optimization, the selected model still produces false positives and false negatives, limiting its reliability for operational decision-making.

- **Feature Constraints:** Model predictions are restricted to the selected 16 predictors and may not capture all external factors influencing flight delays.

- **Generalizability:** The models have not been validated using independent real-world airline datasets.

- **Operational Deployment:** Further validation, probability calibration, decision-threshold optimization and continuous performance monitoring would be required before practical deployment.

## 13. Technologies Used

Python | Pandas | NumPy | Matplotlib | Seaborn | Scikit-learn

