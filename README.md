# ITDAA4 Data Mining Project (B23)

A data mining project completed for the ITDAA4 (Data Mining) module, covering data cleaning, exploratory data analysis, clustering, and classification on student performance data.

## Overview

The notebook works through two related student performance datasets, progressing from data cleaning through unsupervised and supervised learning:

1. **Exploratory Analysis & Clustering** — `student_performance.csv`
2. **Predictive Classification** — `ai_impact_student_performance_dataset.csv`
3. **SQL Queries** — a set of queries against a `Student_Performance` table

## 1. Exploratory Analysis & Clustering

- Cleaned inconsistent categorical values (e.g. qualification names, yes/no variants in `passed_all_modules`)
- Imputed missing values (tutorial attendance filled with 0; grade average filled with the group median by qualification)
- Visualized the average and distribution of pass rate by qualification (bar chart, boxplot)
- Compared pass-all-modules rates for students who attended tutorials versus those who didn't, overall and by qualification
- Examined tutorial attendance by study level (1st–3rd year)
- Standardized numeric features and reduced them to two components with **PCA** (53.1% / 22.9% explained variance, ~76% cumulative)
- Selected `k=3` clusters via the elbow method and silhouette score (silhouette ≈ 0.504)
- Ran **K-Means** clustering on the PCA-reduced features and visualized clusters by study level
- Profiled each cluster's average feature values
- Exported the cleaned, cluster-labeled dataset to `student_performance_cleaned_with_clusters.csv`

## 2. Predictive Classification

- Loaded a separate 8,000-row dataset on AI tool usage and student performance
- Handled missing values, dropped leakage columns (`final_score`, `performance_category`, `student_id`), and one-hot encoded categorical features
- Split into train/test sets (75/25, stratified) and standardized numeric features
- Selected the top 10 features using `SelectKBest` (ANOVA F-test); top features included last exam score, assignment average, concept understanding score, and AI-related usage metrics
- Trained and compared two classifiers predicting whether a student passed:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.882 | 0.991 | 0.876 | 0.930 |
| Decision Tree (max_depth=5) | 0.869 | 0.982 | 0.868 | 0.922 |

- Visualized confusion matrices and a side-by-side metric comparison for both models
- Compared logistic regression coefficients against decision tree feature importances

## 3. SQL Queries

A short set of SQL queries against a `Student_Performance` table, including: total student count, average pass rate, students who passed all modules, average tutorial attendance by study level, and the qualification with the highest average pass rate.

## Requirements

- Python 3
- pandas, numpy, matplotlib
- scikit-learn

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Data

This notebook expects the following files in the same directory:

- `student_performance.csv`
- `ai_impact_student_performance_dataset.csv`

## Usage

Open `itdaa4_b23_project__.ipynb` in Jupyter and run the cells in order.
