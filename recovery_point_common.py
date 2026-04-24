from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

REQUIRED_BASE_COLUMNS = [
    "ACTIVATED_PLAN_ID",
    "GROUP_ID",
    "COPY_TYPE",
    "SOURCE_SYSTEM",
    "SOURCE_TYPE",
    "SCANNABLE",
    "SCAN_JOB_RESULT_RAW",
    "SCAN_JOB_RESULT",
    "VALIDATION_SUCCESSFUL_RAW",
    "VALIDATION_STATUS",
    "IS_LATEST",
    "IMMUTABLE",
    "MALWARE_ANOMALY_DETECTED",
]

TRAINING_REQUIRED_COLUMNS = REQUIRED_BASE_COLUMNS + ["LABEL"]

FEATURE_COLUMNS = [
    "COPY_TYPE",
    "SOURCE_SYSTEM",
    "SOURCE_TYPE",
    "SCANNABLE",
    "SCAN_RESULT_NORM",
    "VALIDATION_STATUS_NORM",
    "IS_LATEST",
    "IMMUTABLE_FLAG",
    "MALWARE_ANOMALY_DETECTED",
    "SCORE_MALWARE",
    "SCORE_SCANNER",
    "SCORE_VALIDATION",
    "SCORE_LATEST",
    "SCORE_IMMUTABLE",
    "TOTAL_SCORE",
    "DANGER_COUNT",
]

CATEGORICAL_FEATURES = [
    "COPY_TYPE",
    "SOURCE_SYSTEM",
    "SOURCE_TYPE",
    "SCAN_RESULT_NORM",
    "VALIDATION_STATUS_NORM",
    "IMMUTABLE_FLAG",
]

NUMERIC_FEATURES = [
    "SCANNABLE",
    "IS_LATEST",
    "MALWARE_ANOMALY_DETECTED",
    "SCORE_MALWARE",
    "SCORE_SCANNER",
    "SCORE_VALIDATION",
    "SCORE_LATEST",
    "SCORE_IMMUTABLE",
    "TOTAL_SCORE",
    "DANGER_COUNT",
]


def load_yaml(yaml_path: str | Path) -> Dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().upper().replace(" ", "_") for c in out.columns]
    return out


def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + "\nAvailable columns: "
            + ", ".join(df.columns.tolist())
        )


def normalize_bool(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        v = value.strip().upper()
        if v in {"TRUE", "T", "YES", "Y", "1"}:
            return True
        if v in {"FALSE", "F", "NO", "N", "0"}:
            return False
        if v in {"NONE", "NULL", "NAN", "", "UNKNOWN"}:
            return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return np.nan
        if value == 1:
            return True
        if value == 0:
            return False
    return bool(value)


def normalize_validation_status(value):
    if pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    mapping = {
        "VALIDATED": "VALIDATED",
        "VALID": "VALIDATED",
        "READY": "VALIDATED",
        "TRUE": "VALIDATED",
        "SUCCESS": "VALIDATED",
        "SUCCESSFUL": "VALIDATED",
        "INVALID": "INVALID",
        "FALSE": "INVALID",
        "FAILED": "INVALID",
        "FAILURE": "INVALID",
        "ERROR": "INVALID",
        "NOT_VALIDATED": "NOT_VALIDATED",
        "NOTVALIDATED": "NOT_VALIDATED",
        "UNKNOWN": "UNKNOWN",
        "NULL": "UNKNOWN",
        "NONE": "UNKNOWN",
        "NAN": "UNKNOWN",
        "": "UNKNOWN",
    }
    return mapping.get(text, text)


def normalize_scan_result(value, scannable):
    scannable_norm = normalize_bool(scannable)
    if pd.isna(value) or str(value).strip() == "":
        return "NON_SCANNABLE" if scannable_norm is False else "UNKNOWN"
    text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    mapping = {
        "CLEAN": "CLEAN",
        "INCONCLUSIVE": "INCONCLUSIVE",
        "SUSPICIOUS": "SUSPICIOUS",
        "MALICIOUS": "MALICIOUS",
        "INFECTED": "INFECTED",
        "NOT_SCANNED": "NOT_SCANNED",
        "NOTSCANNED": "NOT_SCANNED",
        "NON_SCANNABLE": "NON_SCANNABLE",
        "NONSCANNABLE": "NON_SCANNABLE",
        "FAILED": "FAILED",
        "FAILURE": "FAILED",
        "ERROR": "FAILED",
        "UNKNOWN": "UNKNOWN",
        "NULL": "UNKNOWN",
        "NONE": "UNKNOWN",
        "NAN": "UNKNOWN",
        "": "UNKNOWN",
    }
    return mapping.get(text, text)


def get_bool_score(score_dict: Dict, value: bool, default=0):
    if value is True:
        return score_dict.get(True, score_dict.get("true", score_dict.get("TRUE", default)))
    return score_dict.get(False, score_dict.get("false", score_dict.get("FALSE", default)))


def score_with_yaml(df: pd.DataFrame, rules: Dict) -> pd.DataFrame:
    out = normalize_columns(df)
    validate_columns(out, REQUIRED_BASE_COLUMNS)
    score_cfg = rules["score"]
    danger_cfg = rules["danger_rules"]
    threshold_cfg = rules["threshold"]

    out["SCANNABLE"] = out["SCANNABLE"].apply(normalize_bool)
    out["IS_LATEST"] = out["IS_LATEST"].apply(normalize_bool)
    out["IMMUTABLE"] = out["IMMUTABLE"].apply(normalize_bool)
    out["MALWARE_ANOMALY_DETECTED"] = out["MALWARE_ANOMALY_DETECTED"].apply(normalize_bool)

    out["SCAN_RESULT_NORM"] = out.apply(
        lambda row: normalize_scan_result(row["SCAN_JOB_RESULT"], row["SCANNABLE"]),
        axis=1,
    )
    out["VALIDATION_STATUS_NORM"] = out["VALIDATION_STATUS"].apply(normalize_validation_status)

    out["SCORE_MALWARE"] = np.where(
        out["MALWARE_ANOMALY_DETECTED"] == True,
        score_cfg["malware"]["danger"],
        score_cfg["malware"]["safe"],
    )
    out["SCORE_SCANNER"] = (
        out["SCAN_RESULT_NORM"].map(score_cfg["scanner"]).fillna(score_cfg["scanner"].get("UNKNOWN", 0))
    )
    out["SCORE_VALIDATION"] = (
        out["VALIDATION_STATUS_NORM"].map(score_cfg["validation"]).fillna(score_cfg["validation"].get("UNKNOWN", 0))
    )
    out["SCORE_LATEST"] = np.where(
        out["IS_LATEST"] == True,
        get_bool_score(score_cfg["latest"], True),
        get_bool_score(score_cfg["latest"], False),
    )
    out["IMMUTABLE_FLAG"] = np.where(
        out["IMMUTABLE"].isna(),
        "UNKNOWN",
        np.where(out["IMMUTABLE"] == True, "TRUE", "FALSE"),
    )
    out["SCORE_IMMUTABLE"] = np.select(
        [out["IMMUTABLE_FLAG"].eq("TRUE"), out["IMMUTABLE_FLAG"].eq("FALSE")],
        [get_bool_score(score_cfg["immutable"], True), get_bool_score(score_cfg["immutable"], False)],
        default=score_cfg["immutable"].get("unknown", score_cfg["immutable"].get("UNKNOWN", 0)),
    )
    out["TOTAL_SCORE"] = (
        out["SCORE_MALWARE"]
        + out["SCORE_SCANNER"]
        + out["SCORE_VALIDATION"]
        + out["SCORE_LATEST"]
        + out["SCORE_IMMUTABLE"]
    )
    out["DANGER_MALWARE"] = (out["MALWARE_ANOMALY_DETECTED"] == True).astype(int)
    out["DANGER_SCANNER"] = out["SCAN_RESULT_NORM"].isin(danger_cfg["scanner_danger_values"]).astype(int)
    out["DANGER_VALIDATION"] = out["VALIDATION_STATUS_NORM"].isin(danger_cfg["validation_danger_values"]).astype(int)
    out["DANGER_COUNT"] = out[["DANGER_MALWARE", "DANGER_SCANNER", "DANGER_VALIDATION"]].sum(axis=1)

    def derive_rule_label(row):
        if row["DANGER_COUNT"] >= 2:
            return "AVOID"
        if row["DANGER_COUNT"] == 1:
            return "RISKY"
        if row["TOTAL_SCORE"] >= threshold_cfg["BEST"]:
            return "BEST"
        if row["TOTAL_SCORE"] >= threshold_cfg["GOOD"]:
            return "GOOD"
        if row["TOTAL_SCORE"] >= threshold_cfg["ACCEPTABLE"]:
            return "ACCEPTABLE"
        return "RISKY"

    out["RULE_LABEL"] = out.apply(derive_rule_label, axis=1)
    return out


def build_feature_matrix(scored_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = scored_df.copy()
    features = pd.DataFrame(
        {
            "COPY_TYPE": df["COPY_TYPE"].fillna("UNKNOWN").astype(str).str.upper(),
            "SOURCE_SYSTEM": df["SOURCE_SYSTEM"].fillna("UNKNOWN").astype(str),
            "SOURCE_TYPE": df["SOURCE_TYPE"].fillna("UNKNOWN").astype(str).str.upper(),
            "SCANNABLE": df["SCANNABLE"].fillna(False).astype(bool),
            "SCAN_RESULT_NORM": df["SCAN_RESULT_NORM"].fillna("UNKNOWN").astype(str),
            "VALIDATION_STATUS_NORM": df["VALIDATION_STATUS_NORM"].fillna("UNKNOWN").astype(str),
            "IS_LATEST": df["IS_LATEST"].fillna(False).astype(bool),
            "IMMUTABLE_FLAG": df["IMMUTABLE_FLAG"].fillna("UNKNOWN").astype(str),
            "MALWARE_ANOMALY_DETECTED": df["MALWARE_ANOMALY_DETECTED"].fillna(False).astype(bool),
            "SCORE_MALWARE": df["SCORE_MALWARE"].astype(float),
            "SCORE_SCANNER": df["SCORE_SCANNER"].astype(float),
            "SCORE_VALIDATION": df["SCORE_VALIDATION"].astype(float),
            "SCORE_LATEST": df["SCORE_LATEST"].astype(float),
            "SCORE_IMMUTABLE": df["SCORE_IMMUTABLE"].astype(float),
            "TOTAL_SCORE": df["TOTAL_SCORE"].astype(float),
            "DANGER_COUNT": df["DANGER_COUNT"].astype(int),
        }
    )
    return features[FEATURE_COLUMNS], CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))]),
                NUMERIC_FEATURES,
            ),
        ]
    )
