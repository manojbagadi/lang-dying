"""
Model Training, Cross-Validation Benchmark & Model Serialization
Compares Logistic Regression, Random Forest, HistGradientBoosting, and LightGBM.
Generates full classification metrics, confusion matrices, and exports production pipeline.
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def build_pipeline_and_benchmark(data_csv_path: str, output_dir: str):
    print("=" * 65)
    print(" 🚀 VANISHING VOICES: BENCHMARK & MODEL SELECTION PIPELINE")
    print("=" * 65)

    df = pd.read_csv(data_csv_path)
    print(f"Loaded clean dataset: {len(df)} rows")

    # Define Feature Sets
    numeric_features = [
        'log_speakers',
        'log_speakers_per_country',
        'num_countries',
        'Latitude',
        'Longitude',
        'abs_latitude',
        'nearby_language_density'
    ]
    categorical_features = ['macro_region', 'climate_zone']

    all_features = numeric_features + categorical_features
    X = df[all_features]
    y_binary = df['target_binary']

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    # Candidate Models
    models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'HistGradientBoosting': HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_depth=6,
            class_weight='balanced', random_state=42
        )
    }

    if HAS_LGBM:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            class_weight='balanced', random_state=42, verbose=-1
        )

    # 1. Stratified 5-Fold Cross Validation Benchmark
    print("\n[Phase 1] Running 5-Fold Stratified Cross-Validation on all candidate models...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'balanced_acc': 'balanced_accuracy',
        'f1_macro': 'f1_macro',
        'roc_auc': 'roc_auc',
        'pr_auc': 'average_precision'
    }

    cv_results_summary = {}

    for name, clf in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        scores = cross_validate(pipe, X, y_binary, cv=cv, scoring=scoring, n_jobs=-1)
        
        cv_results_summary[name] = {
            'Accuracy': float(np.mean(scores['test_accuracy'])),
            'Balanced Accuracy': float(np.mean(scores['test_balanced_acc'])),
            'F1 Macro': float(np.mean(scores['test_f1_macro'])),
            'ROC-AUC': float(np.mean(scores['test_roc_auc'])),
            'PR-AUC': float(np.mean(scores['test_pr_auc']))
        }
        print(f"\nModel: {name}")
        print(f"  • Accuracy:          {cv_results_summary[name]['Accuracy']:.4f} (±{np.std(scores['test_accuracy']):.4f})")
        print(f"  • Balanced Accuracy: {cv_results_summary[name]['Balanced Accuracy']:.4f}")
        print(f"  • F1 Macro:          {cv_results_summary[name]['F1 Macro']:.4f}")
        print(f"  • ROC-AUC:           {cv_results_summary[name]['ROC-AUC']:.4f}")
        print(f"  • PR-AUC:            {cv_results_summary[name]['PR-AUC']:.4f}")

    # Determine Champion Model (Ranked by Balanced Accuracy and ROC-AUC)
    champion_name = max(cv_results_summary.keys(), key=lambda k: cv_results_summary[k]['ROC-AUC'] + cv_results_summary[k]['Balanced Accuracy'])
    print(f"\n🏆 CHAMPION MODEL SELECTED: {champion_name}")

    # 2. Holdout Test Set Train & Detailed Evaluation
    print("\n[Phase 2] Evaluating Champion Model on 80/20 Stratified Holdout Test Set...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.20, stratify=y_binary, random_state=42
    )

    champion_clf = models[champion_name]
    champion_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', champion_clf)
    ])

    champion_pipeline.fit(X_train, y_train)
    y_test_pred = champion_pipeline.predict(X_test)
    y_test_proba = champion_pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    test_metrics = {
        'accuracy': float(accuracy_score(y_test, y_test_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, y_test_pred)),
        'precision': float(precision_score(y_test, y_test_pred)),
        'recall': float(recall_score(y_test, y_test_pred)),
        'f1_score': float(f1_score(y_test, y_test_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_test_proba)),
        'pr_auc': float(average_precision_score(y_test, y_test_proba))
    }

    print("\nHoldout Test Set Performance:")
    for k, v in test_metrics.items():
        print(f"  • {k:20s}: {v:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred).tolist()
    cls_report = classification_report(y_test, y_test_pred, target_names=['Safe/Low-Risk', 'Endangered/High-Risk'], output_dict=True)

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Safe/Low-Risk', 'Endangered/High-Risk']))

    # 3. Extract Feature Importances via Permutation Importance
    from sklearn.inspection import permutation_importance
    perm_res = permutation_importance(champion_pipeline, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    raw_imp = {feat: float(perm_res.importances_mean[i]) for i, feat in enumerate(all_features)}
    total_val = sum(max(0.0, v) for v in raw_imp.values()) or 1.0
    feature_importances = {k: round((max(0.0, v) / total_val) * 100, 2) for k, v in sorted(raw_imp.items(), key=lambda item: item[1], reverse=True)}

    print("\nTop Predictive Features (Permutation Importance):")
    for feat, imp in list(feature_importances.items())[:8]:
        print(f"  • {feat:30s}: {imp:.2f}%")

    # 4. Fit Champion Pipeline on Full Data for Production Deployment
    print("\n[Phase 3] Retraining Champion Pipeline on 100% of Cleaned Data for Production...")
    champion_pipeline.fit(X, y_binary)

    # 5. Export Artifacts
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "champion_pipeline.joblib")
    joblib.dump(champion_pipeline, model_save_path)
    print(f"Saved Production Pipeline to: {model_save_path}")

    # Save metadata JSON
    metadata = {
        'champion_model': champion_name,
        'cv_benchmark': cv_results_summary,
        'test_metrics': test_metrics,
        'confusion_matrix': cm,
        'classification_report': cls_report,
        'feature_importances': feature_importances,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'feature_names': all_features
    }

    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved Benchmark Metadata to: {metadata_path}")

    print("\n" + "=" * 65)
    print(" ✅ MODEL TRAINING & BENCHMARK COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_csv = os.path.join(base_dir, "data", "languages_clean.csv")
    models_dir = os.path.join(base_dir, "models")
    build_pipeline_and_benchmark(clean_csv, models_dir)
