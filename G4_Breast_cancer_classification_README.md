# 🩺 Breast Cancer Classification Project

## 📌 Project Overview

This project aims to build a machine learning classification model to predict whether a breast tumor is malignant or benign based on patient and tumor characteristics. The goal is to assist medical professionals in early detection and diagnosis of breast cancer, which improves clinical decision-making and reduces cancer severity.

The dataset used is publicly available on [Kaggle](https://www.kaggle.com/datasets/fatemehmehrparvar/breast-cancer-prediction), and includes medical features such as age, tumor size, number of affected lymph nodes, and metastasis status (the spread of cancer to other tissues).

---

## 📊 Dataset Description

The dataset contains several important features:

| Feature Name         | Description                                  |
|----------------------|----------------------------------------------|
| Age                  | Patient's age                                |
| TumorSize(cm)        | Tumor size in centimeters                    |
| Inv-Nodes            | Number of invasive lymph nodes               |
| Metastasis           | Metastasis status                            |
| Menopause            | Menopause status                             |
| Breast               | Side of the breast affected                  |
| BreastQuadrant       | Location of tumor within the breast          |
| History              | Whether patient had breast cancer history    |
| Diagnosis            | Target variable (Benign or Malignant)        |

Some features were log-transformed (e.g., TumorSize) to normalize skewed distributions.

---

## 🧠 Machine Learning Models Used

Three different classifiers were trained and compared, selected due to:

1. **Logistic Regression**  
   - This is baseline linear classifier with good interpretability
   - Can be regularized to prevent overfitting
   - Good for understanding feature influence

2. **Support Vector Machine (SVM)**  
   - Excellent for high-dimensional classification problems
   - Capable of capturing non-linear relationships using kernel tricks
   - It is robust to outliers with proper tuning

3. **Random Forest Classifier**  
   - Ensemble method using decision trees
   - It handles non-linearities and interactions well
   - Provides feature importance and performs well out-of-the-box

All models were integrated into pipelines with preprocessing, and optimized using **hyperparameter tuning** (`GridSearchCV`).

---

## 🔧 Preprocessing and Pipeline

- Missing values imputed:
  - Numerical: Mean
  - Categorical: Mode
- Standard scaling for numerical features
- One-hot encoding for categorical features
- Log transformation of skewed numeric feature - TumorSize
- Combined using `ColumnTransformer` and `Pipeline` for modularity

---

## 🔍 Model Evaluation

The three selected models were evaluated using **Stratified 5-Fold Cross-Validation** with the following metrics:

| Metric         | Reason for Use |
|----------------|----------------|
| Accuracy       | Assesses general model performance |
| **Recall**     | Most important — minimizes false negatives in cancer diagnosis |
| Precision      | Ensure predictions labeled malignant are actually malignant |
| F1-Score       | Balance between precision and recall |
| **ROC AUC**    | Measures model's ability to rank benign vs malignant tumors |

We also used **Confusion Matrices** for each model to analyze classification errors.

---

## 📈 Results Summary

| Model                  | Accuracy | Precision (Malignant) | Recall (Malignant) | F1 Score (Malignant) | ROC AUC |
| ---------------------- | -------- | --------------------- | ------------------ | -------------------- | ------- |
| Logistic Regression    | 0.8920   | 0.927                 | 0.817              | 0.869                | 0.935   |
| Support Vector Machine | 0.8967   | 0.973                 | 0.785              | 0.869                | 0.920   |
| Random Forest          | 0.8545   | 0.844                 | 0.817              | 0.831                | 0.941   |


While the Support Vector Machine model achieved the highest precision for detecting malignant tumors, the Random Forest model demonstrated a better balance between precision and recall, achieving the highest ROC AUC score and maintaining strong recall and precision. This balance makes Random Forest the ideal model for this breast cancer classification task, as it effectively minimizes false negatives while maintaining a good overall classification performance.

---

## 📉🔍 Feature Correlation and Feature Importance Insights

- **Positive correlations** were observed between these features:
  - **Age**, **Tumor Size**, **Invasive Nodes**, and **Metastasis**
- Clinical interpretation:
  - Older patients tend to present with more advanced tumors
  - Larger tumors are more likely to have spread to nodes and cause metastasis
- These correlations help models learn patterns of cancer progression


Using both **Random Forest's built-in feature importance** and **SHAP (SHapley Additive exPlanations)**, we identified the most influential features in predicting breast cancer malignancy:

1. **Tumor Size (log-transformed)** – Larger tumor sizes strongly correlate with malignancy.
2. **Number of Invasive Nodes** – Higher counts indicate more aggressive cancer progression.
3. **Age** – Older patients tend to show a higher likelihood of malignancy.

These findings align with medical understanding and help validate the model’s learning behavior. SHAP visualizations further enhance interpretability by showing the individual contribution of each feature to predictions.

---
## 🧪 Key Takeaways

- **Recall** is prioritized due to the clinical importance of minimizing false negatives
- All models benefit from preprocessing and hyperparameter tuning
- Feature correlations were found to reflect real-world medical progression
- Random Forest outperforms others on balanced performance and recall

---

## 📂 Project Structure

```
breast-cancer-classification/
│
├── data/                  # Dataset file (csv)
├── notebook/              # Jupyter notebook report
├── presentation           # Presentation slides
└── README.md              # Project documentation
```

---

## 🚀 Future Enhancements

- To deploy the best model with a web-based user interface for clinical usability
- Incorporate more advanced ensemble methods (e.g., XGBoost)
- Evaluate on external test datasets for generalization
-	Integrating time-to-event models such as the Cox Regression model used in survival analysis. For example predicting survival time in breast cancer patients.

---
## Limitations
- The dataset is small, which limits the generisability of the findings.

- The dataset lacks imaging and genomic data, and this limits the depth and dimensionality of the analysis.

- There may be demographic imbalances in the dataset, for instance underrepresented races or age groups. This can introduce bias in model predictions and limit generalizability.

- Feature selection relies on existing structured data, which may omit unstructured clinical notes and physician observations.

- Real-time validation in clinical workflows and integration with electronic health records (EHR) is beyond the current scope.


- The dataset consists of patient records from a single hospital, which can also introduce bias in the results.

---

## 🤝 Acknowledgments

- Dataset by [Fatemeh Mehrparvar on Kaggle](https://www.kaggle.com/datasets/fatemehmehrparvar/breast-cancer-prediction)
- Scikit-learn for model development and evaluation tools

---

## 📜 License

This project is for educational and research purposes only and not intended for clinical use without validation.
