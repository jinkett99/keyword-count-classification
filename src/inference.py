from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, logger
import pandas as pd
import joblib

import joblib
import pandas as pd

def run_inference(threshold=0.38):
    # List file paths
    model_path = MODEL_DIR / 'modelv3.pkl'
    test_data_path = PROCESSED_DATA_DIR / 'test2022.csv'
    output_path = PROCESSED_DATA_DIR / 'ie2022_inferencev2.csv'
    
    # Load the trained Random Forest model
    model = joblib.load(model_path)
    
    # Load the test dataset
    test_data = pd.read_csv(test_data_path)
    
    # Drop the 'UEN' column
    X_test = test_data.drop(columns=['UEN'])
    
    # Run inference on the test dataset
    y_proba = model.predict_proba(X_test)
    
    # Apply the threshold to determine the predicted class
    y_pred = (y_proba[:, 1] >= threshold).astype(int)
    
    # Add the predicted class as a new column
    test_data['pred_class'] = y_pred
    
    # Convert 0 and 1 labels back to 'B' and 'C'
    test_data['pred_class'] = test_data['pred_class'].map({0: 'B', 1: 'C'})
    
    # Output the final dataset to the specified output path
    test_data.to_csv(output_path, index=False)

    
if __name__ == "__main__":
    run_inference()
