import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def detect_outliers(df: pd.DataFrame, num_cols: list) -> dict:
    """
    Detects outliers in numerical columns using the IQR method.
    """
    outlier_summary = {}
    total_rows = len(df)
    
    if total_rows == 0:
        return {}

    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
            
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / total_rows) * 100
        
        if outlier_count > 0:
            outlier_summary[col] = {
                "count": outlier_count,
                "percentage": round(outlier_pct, 2),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "min_outlier": float(outliers.min()),
                "max_outlier": float(outliers.max())
            }
            
    return outlier_summary

def analyze_correlations(df: pd.DataFrame, num_cols: list) -> dict:
    """
    Computes Pearson correlation matrix and extracts top correlation pairs.
    """
    if len(num_cols) < 2:
        return {"matrix": {}, "top_pairs": []}
        
    corr_df = df[num_cols].corr(method='pearson')
    
    # Extract top correlated pairs (absolute value sorted, excluding diagonal)
    pairs = []
    cols = corr_df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            val = corr_df.loc[c1, c2]
            if not pd.isna(val):
                pairs.append({
                    "feature_a": c1,
                    "feature_b": c2,
                    "coefficient": float(round(val, 4)),
                    "abs_coefficient": float(round(abs(val), 4))
                })
                
    pairs = sorted(pairs, key=lambda x: x["abs_coefficient"], reverse=True)
    
    return {
        "matrix": corr_df.to_dict(),
        "top_pairs": pairs[:10]  # Top 10 relationships
    }

def analyze_feature_importance(df: pd.DataFrame, target_col: str, profiling_info: dict) -> dict:
    """
    Determines if target is classification or regression, pre-processes features,
    and runs a RandomForest to estimate feature importances.
    """
    if target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not found in dataset."}
        
    df_clean = df.copy()
    
    # 1. Determine task type (Classification vs. Regression)
    target_series = df_clean[target_col].dropna()
    if len(target_series) == 0:
        return {"error": "Target column has only missing values."}
        
    target_type = "classification"
    unique_vals = target_series.nunique()
    
    # If the target is numeric and has many unique values, treat as regression
    if pd.api.types.is_numeric_dtype(target_series) and unique_vals > 10:
        target_type = "regression"
    elif pd.api.types.is_float_dtype(target_series) and unique_vals > 5:
        target_type = "regression"
        
    # Drop rows where target is null
    df_clean = df_clean.dropna(subset=[target_col])
    
    y = df_clean[target_col]
    X = df_clean.drop(columns=[target_col])
    
    # 2. Identify drop candidates (Text fields, ID-like features, single value features)
    drop_cols = []
    for col in X.columns:
        # Drop columns with all nulls
        if X[col].isna().sum() == len(X):
            drop_cols.append(col)
            continue
            
        # Drop identifier-like column names or text
        col_lower = col.lower()
        if col_lower in ["id", "index", "uuid", "row_id", "serial"] or col_lower.endswith("_id"):
            drop_cols.append(col)
            continue
            
        # Check text cols from profiling
        if col in profiling_info.get("column_types", {}).get("text", []):
            drop_cols.append(col)
            continue
            
        # Drop columns with single value
        if X[col].nunique() <= 1:
            drop_cols.append(col)
            
    X = X.drop(columns=drop_cols)
    
    if len(X.columns) == 0:
        return {
            "task_type": target_type,
            "error": "No valid feature columns left after preprocessing.",
            "importances": {}
        }
        
    # 3. Preprocess X: impute missing values, encode categoricals
    label_encoders = {}
    encoded_cols = []
    
    for col in X.columns:
        col_series = X[col]
        # Impute
        if pd.api.types.is_numeric_dtype(col_series):
            median_val = col_series.median()
            # If median_val is NaN (all values were NaN somehow, though dropped above), fallback to 0
            if pd.isna(median_val):
                median_val = 0
            X[col] = col_series.fillna(median_val)
        else:
            # Categorical encoding
            mode_series = col_series.mode()
            mode_val = mode_series.iloc[0] if len(mode_series) > 0 else "missing"
            X[col] = col_series.fillna(mode_val).astype(str)
            
            # Label encode
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
            encoded_cols.append(col)
            
    # 4. Target encoding if classification
    if target_type == "classification":
        le_y = LabelEncoder()
        y = le_y.fit_transform(y.astype(str))
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    else:
        y = y.astype(float)
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        
    # Fit Model
    try:
        model.fit(X, y)
        importances = model.feature_importances_
        importance_dict = {col: float(imp) for col, imp in zip(X.columns, importances)}
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
        
        return {
            "task_type": target_type,
            "importances": sorted_importance,
            "features_used": list(X.columns)
        }
    except Exception as e:
        return {
            "task_type": target_type,
            "error": f"Error fitting random forest model: {str(e)}"
        }
