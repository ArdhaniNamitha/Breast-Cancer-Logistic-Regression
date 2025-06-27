# Breast-Cancer-Logistic-Regression
A machine learning project using logistic regression to classify malignant and benign breast cancer tumors. Includes data preprocessing, model training, evaluation (confusion matrix, ROC-AUC), and threshold tuning with sigmoid function analysis.

## Objective

The goal is to use logistic regression to build a predictive model that can assist in early detection of breast cancer. By evaluating the relationship between tumor characteristics and diagnosis, this project demonstrates how logistic regression can be used to solve binary classification problems with explainable outputs.

---

## Dataset Description

The dataset consists of 569 instances with 30 numeric features that describe characteristics of cell nuclei in digitized images. The target variable is `diagnosis`, where:
- `M` indicates malignant
- `B` indicates benign

The dataset is available in CSV format and can be accessed from the repository.

**Download Dataset**: [Click to view/download the dataset](data%20(1).csv)

---

## Files Included in the Repository

- `code(Task4).ipynb` — Jupyter Notebook containing all code, visualizations, and model steps.
- `data (1).csv` — The dataset used for training and testing.
- `curve(ROC).png` — ROC curve showing model's ability to distinguish between classes.
- `threshold_tuning.png` — Plot visualizing precision and recall across thresholds.
- `sigmoid_function.png` — Visualization of the sigmoid function used in logistic regression.

---

## Steps Covered in the Project

### 1. Data Preprocessing
- Loaded data using pandas
- Dropped unnecessary columns (`id`, `Unnamed: 32`)
- Encoded the target variable (`diagnosis`) into binary values (`M` → 1, `B` → 0)
- Checked and removed missing/null values

### 2. Feature Preparation
- Separated features (X) and labels (y)
- Performed train-test split (80/20) using `train_test_split`
- Standardized features using `StandardScaler` to improve model performance

### 3. Model Building
- Trained a `LogisticRegression` model from `sklearn.linear_model`
- Used `predict` and `predict_proba` to generate predictions and probabilities on test data

### 4. Model Evaluation
- **Confusion Matrix**: Evaluated true positives, false positives, false negatives, and true negatives
- **Classification Report**: Included metrics like precision, recall, F1-score
- **ROC-AUC Score**: Measured discriminatory ability of the model
- **ROC Curve**: Plotted true positive rate vs. false positive rate to visualize performance

**View ROC Curve**: [ROC Curve](curve(ROC).png)

---

## Threshold Tuning

Rather than using the default 0.5 decision threshold, the project explores how changing the threshold impacts precision and recall. A threshold tuning plot is generated to visualize the trade-off and support better decision-making in applications where false negatives or false positives are more costly.

**View Threshold Tuning Plot**: [Threshold Tuning Plot](threshold_tuning.png)

---

## Understanding the Sigmoid Function

The sigmoid function maps the model's linear outputs to probabilities between 0 and 1. This is critical for logistic regression, as it forms the basis for interpreting predicted probabilities. A plot of the sigmoid function is included to explain its behavior.

**View Sigmoid Function Plot**: [Sigmoid Function Plot](sigmoid_function.png)

---

## Key Takeaways

- Logistic Regression provides a clear, interpretable approach to binary classification problems.
- Proper feature scaling and preprocessing are essential to model effectiveness.
- Precision, recall, and ROC-AUC provide deeper insight than accuracy alone.
- Threshold tuning is a valuable step for improving classification results in domain-specific applications.
- The sigmoid function offers a probabilistic interpretation of model outputs.

---

## Future Enhancements

- Introduce cross-validation for more robust model validation.
- Apply regularization techniques (L1/L2) to manage potential overfitting.
- Compare performance with alternative classifiers (e.g., Random Forest, SVM).
- Use feature importance analysis or SHAP values for better explainability.
- Deploy the model with a web interface for clinical usage simulation.

---

## How to Run

1. Clone this repository.
2. Open `code(Task4).ipynb` in Jupyter Notebook or any compatible IDE.
3. Ensure `data (1).csv` is present in the same directory.
4. Run the notebook cells sequentially to reproduce all results, metrics, and plots.
