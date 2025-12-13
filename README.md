Telco Customer Churn Prediction
Customer churn prediction model using logistic regression on the Telco dataset. Achieves 81% accuracy, 84% ROC AUC, and 62% F1-score for churn class.

📊 Dataset
Source: Telco customer data (7043 rows × 33 columns originally)

Target: Churn Value (0 = No churn, 1 = Churn) - 26% churn rate

Key features: Tenure, Monthly/Total Charges, Contract type, Internet Service, Payment Method, Services (security, backup, streaming), Demographics

🧹 Data Cleaning Pipeline
python
1. Load Excel → 7043 rows
2. Fix Total Charges (string → float, drop 11 NaN rows) → 7032 rows
3. Binary encoding: Yes/No → 0/1 for 13 columns
4. Service encoding: "No internet/phone service" → 0
5. One-hot encoding: Gender, Internet Service, Contract, Payment Method
6. Drop: IDs, geo columns, leaky features (Churn Score, CLTV)
Final: 7032 rows × 24 numeric columns

🤖 Model: Logistic Regression
python
LogisticRegression(max_iter=1000, solver='lbfgs')
All 23 features used (tenure, charges, services, demographics, encoded categoricals)

Train/Test: 80/20 stratified split

📈 Performance Metrics
text
Accuracy:  80.5%  (vs 74% baseline)
Churn Recall: 60.2%  (catches 225/374 churners)
Churn Precision: 64.3%
Churn F1: 62.2%
ROC AUC: 84.3%  ⭐ Strong separation
Confusion Matrix:

text
[[908 125]  ← Non-churn
 [149 225]] ← Churn
🚀 Quick Start
bash
# 1. Clone & install
git clone <repo>
pip install -r requirements.txt

# 2. Run full pipeline
python churn_prediction.py

# 3. View results
jupyter notebook analysis.ipynb
📁 File Structure
text
├── data/
│   └── Telco_customer_churn.xlsx     # Raw data (1.3MB)
├── cleaned_telco_churn.csv          # Cleaned data (7032×24)
├── churn_prediction.py              # Full pipeline
├── model_results.png               # Metrics plots
└── README.md                       # This file
💡 Improvements Applied
✅ Fixed Total Charges parsing (spaces → NaN)

✅ Full numeric encoding (no bool/object issues)

✅ All relevant features engineered

✅ Stratified train/test split

✅ Comprehensive metrics (F1, ROC AUC prioritized for imbalanced churn)

🎯 Next Steps
 class_weight='balanced' → boost churn recall

 Threshold tuning → optimize F1 (target: 65–75%)

 Feature engineering: ARPU = Total Charges/Tenure

 Cross-validation → confirm 84% ROC AUC holds

 Compare Random Forest/XGBoost → squeeze out 2–5% gains

📚 Tech Stack
Python 3.10+

pandas, scikit-learn, numpy, matplotlib

Jupyter Notebook for analysis

Git/GitHub for version control

📝 Requirements
text
pandas==2.1.4
scikit-learn==1.3.2
numpy==1.24.3
matplotlib==3.7.2
openpyxl==3.1.2  # Excel reading
🪪 License
MIT License - feel free to use for portfolio/projects.

Built by [K.V.AJAY CHARIT] | Computer Science Student | ML Portfolio Project
Dec 2025 | RNSIT CSE-E Stream
