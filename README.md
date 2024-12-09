# 📊 Predict Financial Events

This project focuses on predicting financial events using machine learning models applied to financial ratios and accounting variables.

---

## 📂 Project Structure  

1. **Data Preparation**  
   - The dataset `train (2).csv` contains financial and accounting variables.  
   - Data cleaning and preprocessing steps include:  
     - Handling missing values.  
     - Categorizing variables as binary, categorical, numeric, and non-numeric.  

2. **Feature Engineering**  
   - New financial ratios such as ROA, ROE, and others are generated.  
   - Variables are scaled and normalized where appropriate.  

3. **Model Development**  
   - Several machine learning models are implemented, including:  
     - Logistic Regression.  
     - K-Nearest Neighbors (KNN).  
     - Decision Trees.  
     - Random Forest.  
     - Boosting models (XGBoost, AdaBoost).  
   - Hyperparameter tuning is performed using `GridSearchCV` for optimal results.  

4. **Model Evaluation**  
   - Performance metrics include precision, recall, F1-score, and ROC-AUC.  
   - Comparisons across models highlight strengths and weaknesses for each approach.  

5. **Class Balancing**  
   - Imbalanced target variable is addressed using RandomUnderSampler to improve model predictions.  

---

## 🚀 Results Overview  

| Model              | Accuracy  | Recall | F1-Score | ROC-AUC |
|---------------------|-----------|--------|----------|---------|
| Logistic Regression |     72%   |   71%  |    46%   | 70.61%  |
| Log. Reg. otpimized |     71%   |   70%  |    45%   | 70.61%  |
| KNN                 |     68%   |   78%  |    45%   | 77.86%  |
| KNN optimized       |     68%   |   84%  |    45%   | 83.51%  |
| Decision Tree       |     66%   |   83%  |    44%   | 82.79%  |
| DT optimized        |     72%   |   73%  |    46%   | 72.58%  |
| Random Forest       |     74%   |   87%  |    49%   | 87.01%  |
| Rf optimized        |     72%   |   75%  |    46%   | 74.89%  |
| XGBoost             |     70%   |   85%  |    46%   | 84.96%  |
| XGBoost optimized   |     71%   |   75%  |    46%   | 75.44%  |
| AdaBoost            |     72%   |   76%  |    46%   | 75.55%  |
| AdaBoost optimized  |     73%   |   76%  |    47%   | 76.07%  |
 

---

## ⚙️ Requirements  

Install the following libraries to run the notebook:  
- `pandas`  
- `numpy`  
- `seaborn`  
- `matplotlib`  
- `tableone`  
- `scikit-learn`  
- `scipy`  
- `xgboost`  
- `imbalanced-learn`  

Use the following command to install any missing dependencies:  
```bash
pip install pandas numpy seaborn matplotlib tableone scikit-learn scipy xgboost imbalanced-learn
📈 Next Steps
Extend feature engineering to include additional financial ratios or external datasets.
Test alternative balancing techniques, such as SMOTE.
Experiment with deep learning models for improved performance.
📬 Contact
For any inquiries, feel free to reach out at:
Email: your_email@example.com
LinkedIn: linkedin.com/in/yourprofile
