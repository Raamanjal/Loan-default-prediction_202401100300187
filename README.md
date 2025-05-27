# Loan Default Prediction

## Problem Statement

**Objective**: Predict whether a loan applicant will default using classification algorithms and decision trees. Visualize feature importance and evaluate the model using performance metrics.

---

## Introduction

Loan default prediction is a critical task in the banking and financial sector. Financial institutions require robust models to assess the creditworthiness of applicants and minimize risk. This project focuses on building classification models to predict loan defaults based on various applicant attributes.

Goals of the project:
- Train and evaluate classification algorithms.
- Interpret the model using decision tree visualization.
- Identify and analyze important features influencing loan defaults.

---

## Dataset

The dataset contains applicant information with features such as:
- Gender  
- Marital Status  
- Number of Dependents  
- Education  
- Self Employment Status  
- Applicant and Coapplicant Income  
- Loan Amount and Loan Term  
- Credit History  
- Property Area  

**Files used**:
- `train.csv`: Training and validation dataset.
- `test.csv`: Optional dataset for testing (not used in final evaluation).

---

## Methodology

1. **Data Preprocessing**
   - Handled missing values appropriately.
   - Encoded categorical variables using label encoding.
   - Ensured consistent data types and formats.

2. **Model Building**
   - Applied multiple machine learning algorithms:
     - Logistic Regression
     - Decision Tree Classifier
     - Random Forest Classifier

3. **Model Evaluation**
   - Used metrics such as Accuracy, Precision, Recall, and F1-Score.
   - Visualized the decision tree to enhance interpretability.
   - Compared model performance to determine the best approach.

---

## Feature Importance

Key features influencing loan default prediction include:
- Credit History  
- Applicant Income  
- Loan Amount  
- Property Area  

These were identified using the trained decision tree model.

---

## Model Performance

| Model                 | Accuracy |
|-----------------------|----------|
| Logistic Regression   | 78.86%   |
| Decision Tree         | 77.24%   |
| Random Forest         | 75.61%   |

_Logistic Regression performed the best in terms of accuracy._

---

## Decision Tree Visualization

The decision tree provides a clear visual of the model's decision-making process using conditions based on input features.

*(Include or link to the image file of the tree: `decision_tree_visualization.png`)*

---

## Output Samples

Sample predictions on the test set with actual vs predicted labels help evaluate misclassifications and model effectiveness.

---

## Code

All source code is available in the Jupyter Notebook:  
`Loan_Default_Prediction.ipynb`

The notebook includes:
- Data loading and preprocessing
- Model training
- Evaluation metrics
- Visualizations

---

## References and Credits

- Dataset: Provided for project use (`train.csv` and `test.csv`)
- Libraries and tools:
  - Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
  - Jupyter Notebook
- Documentation:
  - Scikit-learn: https://scikit-learn.org/
  - Pandas: https://pandas.pydata.org/
