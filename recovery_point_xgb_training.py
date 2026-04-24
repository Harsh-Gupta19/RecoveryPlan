from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from recovery_point_common import (
    TRAINING_REQUIRED_COLUMNS,
    build_feature_matrix,
    build_preprocessor,
    load_yaml,
    normalize_columns,
    score_with_yaml,
    validate_columns,
)

RANDOM_STATE = 42


def build_xgb_pipeline(preprocessor, params: Dict, num_classes: int) -> Pipeline:
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
        **params,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def run_kfold_model_selection(X_train, y_train, preprocessor, num_classes: int) -> Tuple[Dict, pd.DataFrame]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    param_grid = {
        "n_estimators": [100, 250],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.85],
        "colsample_bytree": [0.85],
        "min_child_weight": [1, 3],
        "reg_lambda": [1.0],
    }
    rows = []
    best_params = None
    best_score = -1
    for params in ParameterGrid(param_grid):
        macro_scores = []
        weighted_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            pipe = build_xgb_pipeline(preprocessor, params, num_classes)
            pipe.fit(X_train.iloc[train_idx], y_train[train_idx])
            pred = pipe.predict(X_train.iloc[val_idx])
            macro_scores.append(f1_score(y_train[val_idx], pred, average="macro"))
            weighted_scores.append(f1_score(y_train[val_idx], pred, average="weighted"))
        row = dict(params)
        row["cv_f1_macro"] = float(np.mean(macro_scores))
        row["cv_f1_weighted"] = float(np.mean(weighted_scores))
        rows.append(row)
        if row["cv_f1_macro"] > best_score:
            best_score = row["cv_f1_macro"]
            best_params = params
    results = pd.DataFrame(rows).sort_values(["cv_f1_macro", "cv_f1_weighted"], ascending=False).reset_index(drop=True)
    return best_params, results


def evaluate_split(name, y_true, y_pred, label_encoder):
    metrics = {
        "split": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }
    print(f"\n{name} metrics")
    print("-" * 60)
    for k, v in metrics.items():
        if k != "split":
            print(f"{k}: {v:.4f}")
    print("\nClassification report")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, digits=4))
    print("Confusion matrix")
    print(confusion_matrix(y_true, y_pred))
    return metrics


def build_prediction_output(scored_df, X_test, y_test, test_pred, final_pipeline, label_encoder):
    base_cols = [
        "ACTIVATED_PLAN_ID",
        "GROUP_ID",
        "COPY_TYPE",
        "SOURCE_SYSTEM",
        "SOURCE_TYPE",
        "SCAN_RESULT_NORM",
        "VALIDATION_STATUS_NORM",
        "IS_LATEST",
        "IMMUTABLE_FLAG",
        "MALWARE_ANOMALY_DETECTED",
        "RULE_LABEL",
        "TOTAL_SCORE",
        "DANGER_COUNT",
    ]
    base_cols = [c for c in base_cols if c in scored_df.columns]
    out = scored_df.loc[X_test.index, base_cols].copy()
    out["ACTUAL_LABEL"] = label_encoder.inverse_transform(y_test)
    out["PREDICTED_LABEL"] = label_encoder.inverse_transform(test_pred)
    proba = final_pipeline.predict_proba(X_test)
    proba_df = pd.DataFrame(proba, columns=[f"PROBA_{c}" for c in label_encoder.classes_], index=X_test.index)
    out = pd.concat([out, proba_df], axis=1)
    sort_cols = [c for c in ["GROUP_ID", "PROBA_BEST", "TOTAL_SCORE"] if c in out.columns]
    if sort_cols:
        ascending = [True if c == "GROUP_ID" else False for c in sort_cols]
        out = out.sort_values(sort_cols, ascending=ascending)
    return out.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel_path", required=True)
    parser.add_argument("--sheet_name", default="Recovery Points Training Data")
    parser.add_argument("--yaml_path", required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--fast", action="store_true", help="Use a smaller parameter grid for faster demo runs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rules = load_yaml(args.yaml_path)
    raw_df = pd.read_excel(args.excel_path, sheet_name=args.sheet_name)
    train_df = normalize_columns(raw_df)
    validate_columns(train_df, TRAINING_REQUIRED_COLUMNS)

    scored_df = score_with_yaml(train_df, rules)
    X, categorical_cols, numeric_cols = build_feature_matrix(scored_df)
    y = scored_df["LABEL"].astype(str).str.upper()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Training shape:", train_df.shape)
    print("Target classes:", label_encoder.classes_.tolist())
    print("Target distribution:")
    print(y.value_counts())
    print("Rule-label distribution:")
    print(scored_df["RULE_LABEL"].value_counts())

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_encoded, test_size=0.15, stratify=y_encoded, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=RANDOM_STATE
    )
    print("Split shapes:", X_train.shape, X_val.shape, X_test.shape)

    preprocessor = build_preprocessor()

    if args.fast:
        best_params = {
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
        }
        cv_results = pd.DataFrame([best_params | {"cv_f1_macro": np.nan, "cv_f1_weighted": np.nan}])
    else:
        best_params, cv_results = run_kfold_model_selection(X_train, y_train, preprocessor, len(label_encoder.classes_))

    print("Best params:")
    print(json.dumps(best_params, indent=2))

    val_pipeline = build_xgb_pipeline(preprocessor, best_params, len(label_encoder.classes_))
    val_pipeline.fit(X_train, y_train)
    val_pred = val_pipeline.predict(X_val)
    val_metrics = evaluate_split("Validation", y_val, val_pred, label_encoder)

    final_pipeline = build_xgb_pipeline(preprocessor, best_params, len(label_encoder.classes_))
    final_pipeline.fit(X_trainval, y_trainval)
    test_pred = final_pipeline.predict(X_test)
    test_metrics = evaluate_split("Test", y_test, test_pred, label_encoder)

    prediction_df = build_prediction_output(scored_df, X_test, y_test, test_pred, final_pipeline, label_encoder)

    bundle = {
        "pipeline": final_pipeline,
        "label_encoder": label_encoder,
        "rules": rules,
        "best_params": best_params,
        "feature_columns": list(X.columns),
        "training_columns": list(train_df.columns),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    model_path = out_dir / rules.get("artifacts", {}).get("model_file", "recovery_point_xgb_model.joblib")
    pred_path = out_dir / rules.get("artifacts", {}).get("test_predictions_file", "recovery_point_test_predictions.xlsx")
    joblib.dump(bundle, model_path)
    with pd.ExcelWriter(pred_path, engine="openpyxl") as writer:
        prediction_df.to_excel(writer, sheet_name="test_predictions", index=False)
        cv_results.to_excel(writer, sheet_name="cv_results", index=False)
        pd.DataFrame([val_metrics, test_metrics]).to_excel(writer, sheet_name="metrics", index=False)
    print("Saved model:", model_path.resolve())
    print("Saved predictions:", pred_path.resolve())


if __name__ == "__main__":
    main()
