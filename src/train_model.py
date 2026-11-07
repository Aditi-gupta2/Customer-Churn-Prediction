"""
Model training, evaluation, comparison, and serialization module for ChurnGuard-AI.
Evaluates Logistic Regression, Random Forest, and Gradient Boosting.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

from data_cleaning import prepare_pipeline_data

def train_and_evaluate_models():
    print("Preparing dataset...")
    data = prepare_pipeline_data()
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)
    }
    
    results = {}
    fitted_models = {}
    
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1_score': round(float(f1), 4),
            'roc_auc': round(float(auc), 4),
            'confusion_matrix': cm
        }
        fitted_models[name] = model
        print(f"[{name}] AUC: {auc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Acc: {acc:.4f}")
        
    # Best model selection by ROC-AUC and Recall (balanced for retention)
    # Gradient Boosting / Random Forest usually highest AUC
    best_model_name = max(results, key=lambda k: results[k]['roc_auc'])
    best_model = fitted_models[best_model_name]
    print(f"\nChampion Model Selected: {best_model_name}")
    
    # Feature Importance analysis
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_[0])
    else:
        importances = np.ones(len(feature_names))
        
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Save artifacts
    joblib.dump(best_model, 'models/churn_model.pkl')
    joblib.dump(fitted_models['Logistic Regression'], 'models/logistic_regression.pkl')
    joblib.dump(fitted_models['Random Forest'], 'models/random_forest.pkl')
    joblib.dump(data['scaler'], 'models/scaler.pkl')
    
    metadata = {
        'champion_model': best_model_name,
        'models_performance': results,
        'feature_names': feature_names,
        'numeric_cols': data['numeric_cols'],
        'top_features': feat_imp.head(10).to_dict(orient='records')
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    # Save test set with predictions for instant dashboard demonstration
    test_eval_df = data['raw_test_df'].copy()
    test_eval_df['actual_churn'] = y_test.values
    test_eval_df['churn_probability'] = best_model.predict_proba(X_test)[:, 1]
    test_eval_df['predicted_churn'] = (test_eval_df['churn_probability'] >= 0.5).astype(int)
    
    # Risk Categories
    def assign_risk(prob):
        if prob >= 0.65:
            return 'High Risk'
        elif prob >= 0.35:
            return 'Medium Risk'
        else:
            return 'Low Risk'
            
    test_eval_df['risk_tier'] = test_eval_df['churn_probability'].apply(assign_risk)
    test_eval_df.to_csv('models/scored_test_customers.csv', index=False)
    
    print("\nModels and metadata saved successfully to models/")
    return metadata

if __name__ == '__main__':
    train_and_evaluate_models()

