import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
)

from preprocessing import preprocess_data, build_feature_pipeline

def evaluate_predictions(y_true, y_pred, y_scores=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    if y_scores is not None:
        roc = roc_auc_score(y_true, y_scores)
        pr = average_precision_score(y_true, y_scores)
    else:
        roc = np.nan
        pr = np.nan
        
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
        "ROC-AUC": roc, "PR-AUC": pr, "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }

def main():
    print("Loading data...")
    df = pd.read_csv('data/fake_job_postings.csv')
    X, y = preprocess_data(df)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grids = {
        "Logistic Regression": {
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "params": {
                "classifier__C": [0.1, 0.5, 1, 2, 5],
                "classifier__class_weight": ["balanced"]
            }
        },
        "Linear SVM": {
            "model": LinearSVC(random_state=42, max_iter=2000, dual=False),
            "params": {
                "classifier__C": [0.1, 0.5, 1, 2, 5],
                "classifier__class_weight": ["balanced"]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 20],
                "classifier__min_samples_leaf": [1, 2],
                "classifier__class_weight": ["balanced"]
            }
        }
    }
    
    os.makedirs('models/tuned', exist_ok=True)
    tuning_results = []
    threshold_results = []
    final_results = []
    
    for name, config in grids.items():
        print(f"\n{'='*40}")
        print(f"--- Tuning {name} ---")
        print(f"{'='*40}")
        
        pipeline = Pipeline([
            ('preprocessor', build_feature_pipeline()),
            ('classifier', config['model'])
        ])
        
        # 1. GridSearchCV for hyperparameter tuning
        print("Running GridSearchCV on training data...")
        search = GridSearchCV(
            pipeline, 
            param_grid=config['params'], 
            cv=cv, 
            scoring='f1',
            n_jobs=1,
            refit=True
        )
        search.fit(X_train, y_train)
        
        best_pipeline = search.best_estimator_
        best_params = search.best_params_
        best_cv_f1 = search.best_score_
        print(f"Best Params: {best_params}")
        print(f"Best CV F1 (fraudulent class): {best_cv_f1:.4f}")
        
        tuning_results.append({
            "Model": name,
            "Best_CV_F1": best_cv_f1,
            "Best_Params": str(best_params)
        })
        
        # 2. Threshold Analysis using cross_val_predict
        print("Performing threshold analysis on CV predictions...")
        if hasattr(best_pipeline.named_steps['classifier'], "predict_proba"):
            method = "predict_proba"
            y_scores_cv = cross_val_predict(best_pipeline, X_train, y_train, cv=cv, method=method, n_jobs=1)[:, 1]
            thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        else:
            method = "decision_function"
            y_scores_cv = cross_val_predict(best_pipeline, X_train, y_train, cv=cv, method=method, n_jobs=1)
            # Standard probability-like thresholds + 0.0 baseline for decision function
            thresholds = [-0.5, 0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
            
        pr_auc_cv = average_precision_score(y_train, y_scores_cv)
        print(f"CV PR-AUC: {pr_auc_cv:.4f}")
        
        best_thresh = 0.5 if method == "predict_proba" else 0.0
        best_thresh_f1 = 0
        
        for t in thresholds:
            y_pred_cv = (y_scores_cv >= t).astype(int)
            prec = precision_score(y_train, y_pred_cv, zero_division=0)
            rec = recall_score(y_train, y_pred_cv, zero_division=0)
            f1 = f1_score(y_train, y_pred_cv, zero_division=0)
            cm = confusion_matrix(y_train, y_pred_cv)
            fp = cm[0, 1]
            fn = cm[1, 0]
            
            threshold_results.append({
                "Model": name,
                "Threshold": t,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "FP": fp,
                "FN": fn
            })
            
            # Select best threshold based on highest F1
            if f1 > best_thresh_f1:
                best_thresh_f1 = f1
                best_thresh = t
                
        print(f"Selected Best Threshold based on CV F1: {best_thresh} (CV F1: {best_thresh_f1:.4f})")
        
        # 3. Final Untouched Test Evaluation
        print(f"Evaluating selected configuration on untouched test set...")
        if method == "predict_proba":
            y_scores_test = best_pipeline.predict_proba(X_test)[:, 1]
        else:
            y_scores_test = best_pipeline.decision_function(X_test)
            
        y_pred_test = (y_scores_test >= best_thresh).astype(int)
        
        res_test = evaluate_predictions(y_test, y_pred_test, y_scores_test)
        res_test["Model"] = name + " (Tuned)"
        res_test["Threshold"] = best_thresh
        final_results.append(res_test)
        
        # 4. Save best pipeline
        model_path = f"models/tuned/{name.lower().replace(' ', '_')}_tuned.joblib"
        joblib.dump(best_pipeline, model_path)
        print(f"Saved tuned model to {model_path}")

    # Save to CSV
    pd.DataFrame(tuning_results).to_csv('models/tuning_results.csv', index=False)
    pd.DataFrame(threshold_results).to_csv('models/threshold_results.csv', index=False)
    pd.DataFrame(final_results).to_csv('models/tuned_model_comparison.csv', index=False)
    
    print("\n" + "="*50)
    print("TUNING & TEST SET EVALUATION COMPLETE")
    print("="*50)
    
    try:
        baseline_df = pd.read_csv('models/model_comparison.csv')
        baseline_df['Version'] = 'Baseline'
        
        tuned_df = pd.DataFrame(final_results)
        tuned_df['Version'] = 'Tuned'
        # Remove " (Tuned)" from Model name to group them cleanly
        tuned_df['Model'] = tuned_df['Model'].str.replace(' (Tuned)', '', regex=False)
        
        cols = ['Model', 'Version', 'Precision', 'Recall', 'F1', 'PR-AUC', 'FP', 'FN']
        
        comp_df = pd.concat([baseline_df, tuned_df], ignore_index=True)[cols]
        comp_df = comp_df.sort_values(by=["Model", "Version"])
        
        print("\nBASELINE VS TUNED COMPARISON:")
        print(comp_df.to_string(index=False))
        
    except Exception as e:
        print(f"Could not print comparative table: {e}")
        print("\nTuned Results:")
        print(pd.DataFrame(final_results).to_string())

if __name__ == '__main__':
    main()
