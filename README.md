# Credit Card Fraud Detection

## Business Problem

Credit card fraud is a growing concern in the financial sector, costing businesses and consumers billions of dollars annually. Fraudulent transactions often blend in with legitimate ones, making detection a complex task. Traditional rule-based fraud detection systems struggle to adapt to evolving fraud patterns, leading to false positives and missed fraudulent activities.

This section aims to develop a machine learning model using **Logistic Regression, RandomForest, XGBoost and LightGBM** to accurately detect fraudulent credit card transactions. After selecting the best model between them; we strive to enhance model performance, improve detection accuracy, and minimize false alarms by leveraging advanced hyperparameter tuning and GPU acceleration.

🎯 **Objectives:**
* Firstly, every process will be functionalized.
* Thoroughly analyze the dataset before the feature engineering and building a model.
* Preprocess the dataset via Feature engineering methods.
* Build a fraud detection model using Logistic Regression, RandomForest, XGBoost and LightGBM for classification.
* Select the best model and optimize model performance through GridSearchCV hyperparameter tuning.
* Evaluate results using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.
* Identify key features influencing fraud detection using feature importance analysis.

By implementing an efficient and well-tuned fraud detection model, this project aims to assist financial institutions in minimizing fraudulent activities, reducing financial losses, and improving security for consumers.

## Dataset Story

The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

**It contains only numerical input variables which are the result of a PCA transformation.** Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. **Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.**

Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.
