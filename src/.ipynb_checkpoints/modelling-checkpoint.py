from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, logger
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from scipy.stats import randint
import argparse
import joblib
import os

def train_eval_model(): 
    train_path = PROCESSED_DATA_DIR / 'train.csv'
    train = pd.read_csv(train_path)

    # Separate features and target variable
    X = train.drop(columns=['UEN', 'TRUE_IE'])
    y = train['TRUE_IE']
    # Convert 'B' to 0 and 'C' to 1
    y = y.map({'B': 0, 'C': 1})

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define the model
    model = RandomForestClassifier(random_state=42)

    # Define the hyperparameters to tune
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10)
    }

    # Perform Random Search with Cross-Validation
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=StratifiedKFold(n_splits=5), scoring='roc_auc', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)

    # Get the best model from Random Search
    best_model = random_search.best_estimator_

    # Make predictions on the test set
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    # Set the classification threshold for class "C" labels
    threshold = 0.35
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1_result = f1_score(y_test, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Print evaluation metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 score: {f1_result:.4f}')
    print(f'ROC-AUC: {roc_auc:.4f}')

    # Print confusion matrix and classification report
    conf_matrix = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    
    # Visualize the confusion matrix
    plt.figure(figsize=(8, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=['B', 'C'], yticklabels=['B', 'C'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    print('Classification Report:')
    print(classification_report(y_test, y_pred, target_names=['B', 'C']))
    
    # Combine the train and test sets for final training
    X_combined = pd.concat([X_train, X_test])
    y_combined = pd.concat([y_train, y_test])

    # Re-train the best model on the combined dataset
    best_model.fit(X_combined, y_combined)

    # Save the re-trained best model to the specified path
    model_path = MODEL_DIR / 'modelv3.pkl'
    joblib.dump(best_model, model_path)
        
def train_eval_model_xgb():
    train_path = PROCESSED_DATA_DIR / 'train.csv'
    train = pd.read_csv(train_path)

    # Separate features and target variable
    X = train.drop(columns=['UEN', 'TRUE_IE'])
    y = train['TRUE_IE']
    # Convert 'B' to 0 and 'C' to 1
    y = y.map({'B': 0, 'C': 1})

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define the model
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

    # Define the hyperparameters to tune
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_child_weight': randint(1, 10),
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
    }

    # Perform Random Search with Cross-Validation
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=StratifiedKFold(n_splits=5), scoring='roc_auc', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)

    # Get the best model from Random Search
    best_model = random_search.best_estimator_

    # Make predictions on the test set
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    # Set the classification threshold for class "C" labels
    threshold = 0.35
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1_result = f1_score(y_test, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Print evaluation metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 score: {f1_result:.4f}')
    print(f'ROC-AUC: {roc_auc:.4f}')

    # Print confusion matrix and classification report
    conf_matrix = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    
    # Visualize the confusion matrix
    plt.figure(figsize=(8, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=['B', 'C'], yticklabels=['B', 'C'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    print('Classification Report:')
    print(classification_report(y_test, y_pred, target_names=['B', 'C']))
    
    # Combine the train and test sets for final training
    X_combined = pd.concat([X_train, X_test])
    y_combined = pd.concat([y_train, y_test])

    # Re-train the best model on the combined dataset
    best_model.fit(X_combined, y_combined)

    # Save the re-trained best model to the specified path
    model_path = MODEL_DIR / 'xgbmodelv1.pkl'
    joblib.dump(best_model, model_path)