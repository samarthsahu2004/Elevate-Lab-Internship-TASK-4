# Elevate-Lab-Internship-TASK-4
Breast Cancer Classification using Logistic Regression 🩺
This project demonstrates the implementation of a Logistic Regression model to classify breast cancer tumors as either malignant or benign. It covers the complete machine learning workflow from data preprocessing and model training to detailed performance evaluation.

📖 Overview
The primary goal is to build a reliable binary classifier for medical diagnosis. This project showcases:

Data cleaning and preparation for a real-world dataset.

The importance of feature scaling for logistic regression.

Training a robust classification model.

In-depth model evaluation using a confusion matrix, precision, recall, and the ROC-AUC score, with a focus on metrics that are critical in a healthcare context.

📊 Dataset
This project uses the Breast Cancer Wisconsin (Diagnostic) Dataset from the UC Irvine Machine Learning Repository, made available on Kaggle. It contains 30 numeric features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

Source: Breast Cancer Wisconsin (Diagnostic) Data Set on Kaggle

Target Variable: diagnosis (Malignant = 1, Benign = 0)

🛠️ Technologies & Libraries Used
The project is implemented in Python 3 and utilizes the following libraries:

Scikit-learn: For model training, feature scaling, and evaluation metrics.

Pandas: For loading and manipulating the dataset.

Matplotlib & Seaborn: For creating visualizations like the confusion matrix and ROC curve.

NumPy: For numerical computations.

kagglehub: For programmatically downloading the dataset.

🚀 Setup & Usage
To get this project running on your local machine, follow these steps.

1. Clone the repository:

Bash

git clone https://github.com/your-username/breast-cancer-classifier.git
cd breast-cancer-classifier
2. Install dependencies:
It is recommended to use a virtual environment.

Bash

pip install scikit-learn pandas matplotlib seaborn numpy kagglehub
(Note: You may need to set up your Kaggle API credentials for kagglehub to work. See the Kagle documentation for instructions.)

3. Execute the script:
Run the main Python file from your terminal. The script will handle data download, training, evaluation, and will display the resulting plots.

Bash

python main.py
📈 Results & Conclusion
The trained logistic regression model demonstrated excellent performance in classifying tumors.

Accuracy: The model achieved an overall accuracy of 97% on the test set.

Recall for Malignant Class: The model correctly identified 95% of all malignant tumors (Recall = 0.95). This is a critical metric, as minimizing false negatives (missed cancers) is the top priority in a clinical setting.

AUC Score: The model achieved an Area Under the ROC Curve (AUC) of 0.99, indicating an outstanding ability to distinguish between malignant and benign cases.

Conclusion: The results show that even a relatively simple model like logistic regression can be highly effective for this classification task when the data is properly preprocessed. The high recall and AUC scores suggest that this model can serve as a reliable tool to aid in medical diagnostics. The project also highlights the importance of choosing the right evaluation metric based on the problem context.
