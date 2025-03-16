# Keyword Count Classification

## Overview
This project explores keyword count classification using machine learning models. Due to confidentiality rules, the use case is masked as it is conducted for a private organization. The solution involves re-training a new ML model optimized on ABC data.

## Solution Approach
1. **Extract ABC Data** – Obtain 2023 data, preferably including 2022, to allow separate year analysis.
2. **Train ML Models** – Train a Random Forest (RF) classifier and XGBoost model on extracted datasets.
3. **Evaluate Model Performance** – Obtain evaluation metrics such as Precision, Recall, F1-score, and ROC-AUC.
4. **Run Inference** – Apply the trained models on 2022 & 2023 base datasets (Hire2022, Hire2023).
5. **Review Class Distributions** – Analyze the distribution of firms classified as 'B' (selectively hiring) and 'C' (actively hiring).

---

## Storyline & Steps Taken
### **Title:** Investigating Distribution of Firms that are Selectively Hiring vs Actively Hiring (Hire2023)

### **Background**
- The team reviewed Hire2023 results and identified inconsistencies in the distribution of "actively hiring" websites.
- Nearly 50% of the websites were classified as "actively hiring" in 2023, prompting an in-depth ML and analytics lifecycle review to understand the trend.

### **Overall Review Process**
1. **Review ML Model Performance**
   - The initial RF classifier showed similar false negative (FN) and false positive (FP) rates, suggesting the model did not overestimate class "C" labels.
   - The 2021 ML model lacked robustness and generalizability to 2022 data, with low precision (0.48) and recall (0.59) for classifying "C" labels. The F1-score was approximately 53%.

2. **Insights from ABC Team**
   - ABC team feedback indicated that firms classified as "actively hiring" were, in reality, "selectively hiring."

3. **Manual Review of Misclassified Websites**
   - A closer review revealed cases where firms predicted as "C" were actually "B" (selectively hiring).

4. **Re-training a New RF Classifier on 2022 ABC Data**
   - **Dataset Preparation:**
     - Training set size: 2,872 records.
     - Class distribution (B:C): 60:40 (minimal class imbalance).
   - **Results:**
     - Improved classification metrics for class "C," with an F1-score of 60%.
     - High recall (~71%), indicating better model performance in capturing true positive "C" labels.
   - **Visualizing Outcomes:**
     - Charts reflecting improved classification performance.

5. **Re-run Imputation Steps** (for internal reference).

6. **Present New Findings** – Summarize insights and highlight model improvements.

---

## Repository Structure
```
├── notebooks/
│   ├── 3.01-jk-modelling.ipynb  # Notebook importing modelling and inference scripts
├── src/
│   ├── dataset.py               # Preprocessing, creating train and test sets
│   ├── modelling.py             # Model training and evaluation (RF, XGBoost)
│   ├── inference.py             # Running inference on new datasets
├── README.md                    # Project documentation
```

## Running the Pipeline
The modelling and inference scripts are located in the `src` folder and were imported into the following notebook for execution:
```bash
notebooks/3.01-jk-modelling.ipynb
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss proposed modifications.
