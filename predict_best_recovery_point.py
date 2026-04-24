from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from recovery_point_common import REQUIRED_BASE_COLUMNS, build_feature_matrix, normalize_columns, score_with_yaml, validate_columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--excel_path", required=True)
    parser.add_argument("--sheet_name", default="Test Prediction Data")
    parser.add_argument("--group_id", required=True)
    parser.add_argument("--output_path", default="best_recovery_point_prediction.xlsx")
    args = parser.parse_args()

    bundle = joblib.load(args.model_path)
    pipeline = bundle["pipeline"]
    label_encoder = bundle["label_encoder"]
    rules = bundle["rules"]

    raw_df = pd.read_excel(args.excel_path, sheet_name=args.sheet_name)
    df = normalize_columns(raw_df)
    validate_columns(df, REQUIRED_BASE_COLUMNS)

    group_df = df[df["GROUP_ID"].astype(str) == str(args.group_id)].copy()
    if group_df.empty:
        available = df["GROUP_ID"].dropna().astype(str).unique().tolist()[:20]
        raise ValueError(f"No records found for GROUP_ID={args.group_id}. Sample available group ids: {available}")

    scored_df = score_with_yaml(group_df, rules)
    X_group, _, _ = build_feature_matrix(scored_df)
    pred_encoded = pipeline.predict(X_group)
    pred_labels = label_encoder.inverse_transform(pred_encoded)
    proba = pipeline.predict_proba(X_group)
    proba_df = pd.DataFrame(proba, columns=[f"PROBA_{c}" for c in label_encoder.classes_], index=scored_df.index)

    output_cols = [
        "GROUP_ID",
        "ACTIVATED_PLAN_ID",
        "COPY_TYPE",
        "SOURCE_SYSTEM",
        "SOURCE_TYPE",
        "SCAN_RESULT_NORM",
        "VALIDATION_STATUS_NORM",
        "IS_LATEST",
        "IMMUTABLE_FLAG",
        "MALWARE_ANOMALY_DETECTED",
        "TOTAL_SCORE",
        "DANGER_COUNT",
        "RULE_LABEL",
    ]
    output = scored_df[output_cols].copy()
    output["PREDICTED_LABEL"] = pred_labels
    output = pd.concat([output, proba_df], axis=1)

    sort_cols = []
    ascending = []
    if "PROBA_BEST" in output.columns:
        sort_cols.append("PROBA_BEST")
        ascending.append(False)
    sort_cols.extend(["TOTAL_SCORE", "DANGER_COUNT"])
    ascending.extend([False, True])
    output = output.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    best = output.head(1)

    print("\nAll candidates:")
    print(output.to_string(index=False))
    print("\nBest recovery point:")
    print(best.to_string(index=False))

    output_path = Path(args.output_path)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        output.to_excel(writer, sheet_name="all_candidates", index=False)
        best.to_excel(writer, sheet_name="best_recovery_point", index=False)
    print("\nSaved:", output_path.resolve())


if __name__ == "__main__":
    main()
