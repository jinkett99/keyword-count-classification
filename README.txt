Note: Use-case is masked due to confidentiality rules of task conducted for private organisation. 
Solution to explore – Re-train new ML model optimized on ABC data
	1. Extract ABC data – 2023, preferably 2022 as well (to separate years)
	2. Train RF classifier & XGBoostmodel on datasets
	3. Obtain evaluation metrics.
    4. Run new model on 2022 & 2023 base datasets (IE2022, IE2023)
    5. Review class distributions of 'B' (selectively hiring) & 'C' (actively hiring).
    
Imputations of firms with websites using ABC data. 

--- Storyline & Steps Taken ---
Title: Investigating Distribution of Firms that are Selectively Hiring vs Actively Hiring (Hire2023)
Background information: 
- The team recently reviewed Hire2023 results & found that there were some issues with the distribution of "actively hiring" websites. I.e. Almost one in every two websites are "actively hiring" in 2023. 
- This review aims to dive deep into the ML and analytics lifecycle to investigate the nuances in trends observed (i.e. High proportions of firms with "actively hiring" websites)

Overall review process: 
1. Look into ML RF classifier - Conclusion was that it was unlikely that the model 'overestimated' the proportion of class "C" labels as the FN and FP rates were similar. *Show confusion matrix & classification report. 
   *2021 ML model used was not robust & cannot generalize that well to 2022 data. I.e. Low precision & recall scores (0.48 & 0.59 respectively) for classifying class C labels. F1 score was ~53%
2. Comments from the ABC team that firms tagged with "actively hiring" websites were in fact, not actively hiring but are "selectively hiring" instead. 
3. Eyeball/Review of misclassified "actively hiring" websites (i.e. firms predicted as "C" but in actual fact, was "B" - "selectively hiring") - Conclusion was that there were these websites present.
4. Train a new RF classifier on latest (2022) ABC dataset (ground truth) & re-run inference/predictions on 2022 dataset. *Reflect charts as outcomes. 
   *Dataset preparation: Training set consists of 2872 records. Ratio of B:C was 60:40 (minimal class imbalance).
   *Results: Improved classification metrics for class "C" (F1 score of 60%). Recall was high (~71%) indicating that the model was able to capture most actual positive class "C" labels
5. Re-run imputation steps (for internal info.)
6. Present new findings*
