import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

from preprocessing import preprocess_data, build_feature_pipeline

def evaluate_model(name, y_true, y_pred, y_prob=None, y_decision=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    if y_prob is not None:
        roc = roc_auc_score(y_true, y_prob)
        pr = average_precision_score(y_true, y_prob)
    elif y_decision is not None:
        roc = roc_auc_score(y_true, y_decision)
        pr = average_precision_score(y_true, y_decision)
    else:
        roc = np.nan
        pr = np.nan
        
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    cr = classification_report(y_true, y_pred)
    
    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": roc,
        "PR-AUC": pr,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "ClassificationReport": cr,
        "ConfusionMatrix": cm
    }

def main():
    print("Loading data...")
    df = pd.read_csv('data/fake_job_postings.csv')
    
    print("Preprocessing data...")
    X, y = preprocess_data(df)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42, max_iter=2000, dual=False),
        "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    }
    
    results = []
    
    os.makedirs('models', exist_ok=True)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline([
            ('preprocessor', build_feature_pipeline()),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        y_prob = None
        y_decision = None
        
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_decision = pipeline.decision_function(X_test)
            
        res = evaluate_model(name, y_test, y_pred, y_prob=y_prob, y_decision=y_decision)
        results.append(res)
        
        model_filename = name.lower().replace(' ', '_') + ".joblib"
        model_path = os.path.join('models', model_filename)
        joblib.dump(pipeline, model_path)
        print(f"Saved to {model_path}")

    # Save results to CSV
    metrics_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['ClassificationReport', 'ConfusionMatrix']} for r in results])
    metrics_df.to_csv('models/model_comparison.csv', index=False)
    
    # Print report
    print("\n" + "="*50)
    print("FINAL EVALUATION REPORT")
    print("="*50)
    print(metrics_df.to_string())
    
    for r in results:
        print("\n" + "-"*30)
        print(f"Model: {r['Model']}")
        print(f"Confusion Matrix:\n{r['ConfusionMatrix']}")
        print(f"TN: {r['TN']}, FP: {r['FP']}, FN: {r['FN']}, TP: {r['TP']}")
        print(f"Classification Report:\n{r['ClassificationReport']}")

if __name__ == '__main__':
    main()
